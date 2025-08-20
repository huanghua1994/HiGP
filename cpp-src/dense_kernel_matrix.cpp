#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#ifdef __APPLE__
#include <sys/types.h>
#include <sys/sysctl.h>
#include <mach/mach.h>
#include <mach/mach_host.h>
#else
#include <sys/sysinfo.h>
#endif

#include <unistd.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <omp.h>

#include "kernels/kernels.h"
#include "dense_kernel_matrix.h"
#include "common.h"
#include "cpu_linalg.hpp"
#include "utils.h"

static size_t get_available_memory() {
#ifdef __APPLE__
    vm_size_t page_size;
    vm_statistics64_data_t vm_stat;
    mach_msg_type_number_t host_size = sizeof(vm_stat) / sizeof(natural_t);
    mach_port_t host_port = mach_host_self();
    host_page_size(host_port, &page_size);
    if (host_statistics64(host_port, HOST_VM_INFO64, 
                         (host_info64_t)&vm_stat, &host_size) != KERN_SUCCESS) {
        // Fallback
        return 1024ULL * 1024ULL * 1024ULL;
    }
    uint64_t available_pages = vm_stat.free_count + vm_stat.inactive_count;
    return (size_t)(available_pages * page_size);
#else
    return (size_t)get_avphys_pages() * (size_t)sysconf(_SC_PAGESIZE);
#endif
}

template<typename VT>
static int is_same_c0c1(
    const int n0, const int ld0, const VT *c0, 
    const int n1, const int ld1, const VT *c1, const VT *param
)
{
    if (n0 != n1) return 0;
    int same_c0c1 = 1;
    int dim = (int) param[0];
    for (int i = 0; i < n0; i++)
    {
        VT diff = 0;
        for (int j = 0; j < dim; j++)
        {
            VT diff_i = c0[i + j * ld0] - c1[i + j * ld1];
            diff += diff_i * diff_i;
        }
        if (diff > 0) same_c0c1 = 0;
    }
    return same_c0c1;
}

// Initialize a dense_krnl_mat
void dense_krnl_mat_init(
    const int n0, const int ld0, const void *c0, 
    const int n1, const int ld1, const void *c1, 
    const void *param, const void *dnoise, const int krnl_id, 
    const int val_type, dense_krnl_mat_p *dkmat
)
{
    dense_krnl_mat_p dkmat_ = (dense_krnl_mat_p) malloc(sizeof(dense_krnl_mat_s));
    memset(dkmat_, 0, sizeof(dense_krnl_mat_s));

    int is_double = (val_type == VAL_TYPE_DOUBLE);
    int is_float  = (val_type == VAL_TYPE_FLOAT);
    size_t val_bytes = is_double ? sizeof(double) : sizeof(float);

    int pt_dim = 0;
    if (is_double) pt_dim = (int) ((double *) param)[0];
    if (is_float)  pt_dim = (int) ((float  *) param)[0];

    dkmat_->nrow     = n0;
    dkmat_->ncol     = n1;
    dkmat_->pt_dim   = pt_dim;
    dkmat_->val_type = val_type;
    dkmat_->krnl_id  = krnl_id;
    dkmat_->param    = malloc(4 * val_bytes);
    dkmat_->X        = malloc(n0 * pt_dim * val_bytes);
    dkmat_->Y        = malloc(n1 * pt_dim * val_bytes);
    memcpy(dkmat_->param, param, 4 * val_bytes);
    ASSERT_PRINTF(dkmat_->X != NULL && dkmat_->Y != NULL, "Failed to allocate memory for a dense_krnl_mat struct\n");
    copy_matrix(val_bytes, pt_dim, n0, c0, ld0, dkmat_->X, n0, 1);
    copy_matrix(val_bytes, pt_dim, n1, c1, ld1, dkmat_->Y, n1, 1);

    if (is_float) 
    {
        dkmat_->same_XY = is_same_c0c1<float>(
            n0, ld0, (float *) c0, 
            n1, ld1, (float *) c1, (float *) param
        );
    }
    if (is_double)
    {
        dkmat_->same_XY = is_same_c0c1<double>(
            n0, ld0, (double *) c0, 
            n1, ld1, (double *) c1, (double *) param
        );
    }

    if (dkmat_->same_XY)
    {
        dkmat_->dnoise = malloc(n0 * val_bytes);
        memset(dkmat_->dnoise, 0, n0 * val_bytes);
        if (dnoise != NULL) memcpy(dkmat_->dnoise, dnoise, n0 * val_bytes);
    }

    *dkmat = dkmat_;
}

// Free a dense_krnl_mat struct
void dense_krnl_mat_free(dense_krnl_mat_p *dkmat)
{
    dense_krnl_mat_p dkmat_ = *dkmat;
    if (dkmat_ == NULL) return;
    free(dkmat_->param);
    free(dkmat_->dnoise);
    free(dkmat_->X);
    free(dkmat_->Y);
    free(dkmat_->k_mat);
    free(dkmat_->dl_mat);
    free(dkmat_);
    *dkmat = NULL;
}

// Try to compute K(X, Y, l) and d K(X, Y, l) / d l and store them in dkmat
int dense_krnl_mat_populate(dense_krnl_mat_p dkmat)
{
    if (dkmat == NULL) return 0;
    if (dkmat->k_mat != NULL && dkmat->dl_mat != NULL) return 1;

    int nrow    = dkmat->nrow;
    int ncol    = dkmat->ncol;
    int krnl_id = dkmat->krnl_id;
    size_t val_bytes = (dkmat->val_type == VAL_TYPE_DOUBLE) ? sizeof(double) : sizeof(float);

    krnl_grad_func krnl_grad = NULL;
    get_krnl_grad_func(krnl_id, NULL, NULL, &krnl_grad);

    // Get the available memory size in bytes (cross-platform)
    size_t avail_mem_bytes = get_available_memory();
    size_t kmat_bytes  = nrow * ncol * val_bytes;
    if ((2 * kmat_bytes) <= ((avail_mem_bytes / 10) * 9))  // Use at most 90% of available memory
    {
        dkmat->k_mat  = malloc(kmat_bytes);
        dkmat->dl_mat = malloc(kmat_bytes);
    } else {
        return 0;
    }

    int n_thread = omp_get_max_threads();
    if (omp_get_num_threads() > 1) n_thread = 1;
    krnl_grad_func_omp(
        nrow, nrow, dkmat->X, ncol, ncol, dkmat->Y, 
        krnl_grad, dkmat->param, nrow, 
        1, dkmat->k_mat, 1, dkmat->dl_mat, 
        dkmat->val_type, n_thread
    );
    return 1;
}

template<typename VT>
static void dense_krnl_mat_grad_matmul(
    dense_krnl_mat_p dkmat, const int B_ncol, VT *B, const int ldB, 
    VT *K_B, VT *dKdl_B, VT *dKdf_B, VT *dKds_B, const int ldC
)
{
    int nrow = dkmat->nrow;
    int ncol = dkmat->ncol;
    size_t val_bytes = sizeof(VT);
    int n_thread = omp_get_max_threads();
    if (omp_get_num_threads() > 1) n_thread = 1;

    // 1. Compute K_B0 = k(X, Y, l) * B and dKdl_B0 = (dK / dl) * B
    VT zero_ = 0, one_ = 1;
    if ((dkmat->k_mat != NULL) && (dkmat->dl_mat != NULL))
    {
        xgemm_(
            notrans, notrans, &nrow, &B_ncol, &ncol,
            &one_,  (VT *) dkmat->k_mat,  &nrow, B, &ldB,
            &zero_, K_B,    &ldC
        );
        if (dKdl_B != NULL)
        {
            xgemm_(
                notrans, notrans, &nrow, &B_ncol, &ncol,
                &one_,  (VT *) dkmat->dl_mat, &nrow, B, &ldB,
                &zero_, dKdl_B, &ldC
            );
        }
    } else {
        // On-the-fly matmul
        int nrow = dkmat->nrow;
        int ncol = dkmat->ncol;
        krnl_grad_func krnl_grad = NULL;
        get_krnl_grad_func(dkmat->krnl_id, NULL, NULL, &krnl_grad);
        krnl_grad_matmul_omp(
            nrow, nrow, dkmat->X, ncol, ncol, dkmat->Y,
            krnl_grad, dkmat->param, dkmat->val_type,
            B, ldB, B_ncol, K_B, dKdl_B, ldC, n_thread
        );
    }

    // 2. Compute dKdf_B = 2 * f * K_B0, K_B = f^2 * K_B0, dKdl_B = f^2 * dKdl_B0
    VT *param = (VT *) dkmat->param;
    VT _2f = param[2] * 2;
    VT f2  = param[2] * param[2];
    #pragma omp parallel for num_threads(n_thread) schedule(static)
    for (int j = 0; j < B_ncol; j++)
    {
        VT *K_Bj    = K_B    + j * ldC;
        VT *dKdf_Bj = dKdf_B + j * ldC;
        VT *dKdl_Bj = dKdl_B + j * ldC;
        if (dKdf_B != NULL)
        {
            #pragma omp simd
            for (int i = 0; i < nrow; i++) dKdf_Bj[i] = _2f * K_Bj[i];
        }
        #pragma omp simd
        for (int i = 0; i < nrow; i++) K_Bj[i] *= f2;   
        if (dKdl_B != NULL)
        {
            #pragma omp simd
            for (int i = 0; i < nrow; i++) dKdl_Bj[i] *= f2;
        }
    }

    // 3. If two input point sets are the same, add s * B and 
    //    diag(dnoise) * B to K_B, and compute dKds * B = eye(n) * B
    if (dkmat->same_XY)
    {
        VT *dnoise = (VT *) dkmat->dnoise;
        VT s = param[3];
        #pragma omp parallel for num_threads(n_thread) schedule(static)
        for (int j = 0; j < B_ncol; j++)
        {
            VT *Bj      = B      + j * ldB;
            VT *K_Bj    = K_B    + j * ldC;
            VT *dKds_Bj = dKds_B + j * ldC;
            #pragma omp simd
            for (int i = 0; i < nrow; i++) K_Bj[i] += (s + dnoise[i]) * Bj[i];
            if (dKds_B != NULL) memcpy(dKds_Bj, Bj, nrow * val_bytes);
        }
    } else {
        if (dKds_B == NULL) return;
        #pragma omp parallel for num_threads(n_thread) schedule(static)
        for (int j = 0; j < B_ncol; j++)
        {
            VT *dKds_Bj = dKds_B + j * ldC;
            memset(dKds_Bj, 0, nrow * val_bytes);
        }
    }
}

// Compute C = M * B, M is the dense kernel matrix K(X, Y) or its derivate matrices, 
// B is a dense input matrix, and C is a dense result matrix
void dense_krnl_mat_grad_matmul(
    dense_krnl_mat_p dkmat, const int B_ncol, void *B, const int ldB, 
    void *K_B, void *dKdl_B, void *dKdf_B, void *dKds_B, const int ldC
)
{
    if (dkmat->val_type == VAL_TYPE_DOUBLE)
    {
        dense_krnl_mat_grad_matmul<double>(
            dkmat, B_ncol, (double *) B, ldB, (double *) K_B, 
            (double *) dKdl_B, (double *) dKdf_B, (double *) dKds_B, ldC
        );
    }
    if (dkmat->val_type == VAL_TYPE_FLOAT)
    {
        dense_krnl_mat_grad_matmul<float>(
            dkmat, B_ncol, (float *) B, ldB, (float *) K_B, 
            (float *) dKdl_B, (float *) dKdf_B, (float *) dKds_B, ldC
        );
    }
}

// Compute K * B only, parameters are the same as dense_krnl_mat_grad_matmul()
void dense_krnl_mat_krnl_matmul(
    dense_krnl_mat_p dkmat, const int B_ncol, void *B, const int ldB, 
    void *K_B, const int ldC
)
{
    dense_krnl_mat_grad_matmul(dkmat, B_ncol, B, ldB, K_B, NULL, NULL, NULL, ldC);
}

template<typename VT>
static void dense_krnl_mat_grad_eval(dense_krnl_mat_p dkmat, VT *K, VT *dKdl, VT *dKdf, VT *dKds)
{
    VT *param = (VT *) dkmat->param;
    VT f = param[2], s = param[3];
    VT f2 = f * f;
    int n_thread = omp_get_max_threads();
    if (omp_get_num_threads() > 1) n_thread = 1;

    int dense_populate = dense_krnl_mat_populate(dkmat);
    ASSERT_PRINTF(dense_populate == 1, "Failed to populate dense matrices in %s\n", __FUNCTION__);

    if (K != NULL)
    {
        VT *k_mat  = (VT *) dkmat->k_mat;
        VT *dnoise = (VT *) dkmat->dnoise;
        #pragma omp parallel for num_threads(n_thread) schedule(static)
        for (int i = 0; i < dkmat->nrow * dkmat->ncol; i++) K[i] = f2 * k_mat[i];
        if (dkmat->same_XY)
        {
            #pragma omp simd
            for (int i = 0; i < dkmat->nrow; i++) K[i * dkmat->nrow + i] += s + dnoise[i];
        }
    }

    if (dKdl != NULL)
    {
        VT *dl_mat = (VT *) dkmat->dl_mat;
        #pragma omp parallel for num_threads(n_thread) schedule(static)
        for (int i = 0; i < dkmat->nrow * dkmat->ncol; i++) dKdl[i] = f2 * dl_mat[i];
    }

    if (dKdf != NULL)
    {
        VT *k_mat = (VT *) dkmat->k_mat;
        #pragma omp parallel for num_threads(n_thread) schedule(static)
        for (int i = 0; i < dkmat->nrow * dkmat->ncol; i++) dKdf[i] = 2 * f * k_mat[i];
    }
    
    if (dKds != NULL)
    {
        #pragma omp parallel for num_threads(n_thread) schedule(static)
        for (int i = 0; i < dkmat->nrow * dkmat->ncol; i++) dKds[i] = 0;
        if (dkmat->same_XY)
        {
            #pragma omp simd
            for (int i = 0; i < dkmat->nrow; i++) dKds[i * dkmat->nrow + i] = 1;
        }
    }
}

// Evaluate K = K(X, Y, l, f, s) and d K / d {l, f, s}
void dense_krnl_mat_grad_eval(dense_krnl_mat_p dkmat, void *K, void *dKdl, void *dKdf, void *dKds)
{
    if (dkmat->val_type == VAL_TYPE_DOUBLE)
        dense_krnl_mat_grad_eval<double>(dkmat, (double *) K, (double *) dKdl, (double *) dKdf, (double *) dKds);
    if (dkmat->val_type == VAL_TYPE_FLOAT)
        dense_krnl_mat_grad_eval<float> (dkmat, (float *)  K, (float *)  dKdl, (float *)  dKdf, (float *)  dKds);
}