#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cfloat>
#include <omp.h>

#include "kernels.h"
#include "../common.h"
#include "../utils.h"
#include "../cpu_linalg.hpp"

// Get the krnl_func and/or krnl_grad_func of a kernel by its ID
void get_krnl_grad_func(const int krnl_id, krnl_func *krnl, krnl_func *gkrnl, krnl_grad_func *krnl_grad)
{
    if (krnl_id == KERNEL_ID_PDIST2)
    {
        if (krnl != NULL) *krnl = pdist2_krnl;
        if (gkrnl != NULL) *gkrnl = NULL;
        if (krnl_grad != NULL) *krnl_grad = NULL;
    }
    if (krnl_id == KERNEL_ID_GAUSSIAN)
    {
        if (krnl != NULL) *krnl = gaussian_krnl;
        if (gkrnl != NULL) *gkrnl = gaussian_grad;
        if (krnl_grad != NULL) *krnl_grad = gaussian_krnl_grad;
    }
    if (krnl_id == KERNEL_ID_MATERN32)
    {
        if (krnl != NULL) *krnl = matern32_krnl;
        if (gkrnl != NULL) *gkrnl = matern32_grad;
        if (krnl_grad != NULL) *krnl_grad = matern32_krnl_grad;
    }
    if (krnl_id == KERNEL_ID_MATERN52)
    {
        if (krnl != NULL) *krnl = matern52_krnl;
        if (gkrnl != NULL) *gkrnl = matern52_grad;
        if (krnl_grad != NULL) *krnl_grad = matern52_krnl_grad;
    }
    if (krnl_id == KERNEL_ID_CUSTOM)
    {
        if (krnl != NULL) *krnl = custom_krnl;
        if (gkrnl != NULL) *gkrnl = custom_grad;
        if (krnl_grad != NULL) *krnl_grad = custom_krnl_grad;
    }
}

// Find a * b == n_thread s.t. (m / a) * (n / b) is closest to a square
static void find_2d_decomp(const int m, const int n, const int n_thread, int *a, int *b)
{
    float min_ratio = FLT_MAX;
    int opt_a = 1, opt_b = n_thread;
    for (int i = 1; i <= n_thread; i++)
    {
        int j = n_thread / i;
        if (i * j != n_thread) continue;
        float mr = (float) m / (float) i;
        float nr = (float) n / (float) j;
        float ratio = mr / nr;
        if (ratio < 1.0f) ratio = 1.0f / ratio;
        if (ratio < min_ratio)
        {
            min_ratio = ratio;
            opt_a = i;
            opt_b = j;
        }
    }
    *a = opt_a;
    *b = opt_b;
}

void krnl_func_omp(
    const int n0, const int ld0, const void *c0,
    const int n1, const int ld1, const void *c1,
    krnl_func krnl, const void *param, const int ldm, void *mat, 
    const int val_type, const int n_thread
)
{
    size_t val_bytes = (val_type == VAL_TYPE_DOUBLE) ? sizeof(double) : sizeof(float);
    int n_thread_ = (n_thread <= 0) ? omp_get_max_threads() : n_thread;
    if (omp_get_num_threads() > 1) n_thread_ = 1;
    int nt0, nt1;
    find_2d_decomp(n0, n1, n_thread_, &nt0, &nt1);
    #pragma omp parallel num_threads(n_thread_)
    {
        int tid = omp_get_thread_num();
        int tid0 = tid % nt0, tid1 = tid / nt0;
        int srow = 0, nrow = 0, scol = 0, ncol = 0;
        calc_block_spos_len(n0, nt0, tid0, &srow, &nrow);
        calc_block_spos_len(n1, nt1, tid1, &scol, &ncol);
        const char *c0_ptr = ((char *) c0) + srow * val_bytes;
        const char *c1_ptr = ((char *) c1) + scol * val_bytes;
        size_t mat_displs = (srow * val_bytes) + (scol * ldm * val_bytes);
        char *mat_ptr = ((char * ) mat) + mat_displs;
        krnl(
            nrow, ld0, (const void *) c0_ptr, 
            ncol, ld1, (const void *) c1_ptr, 
            param, ldm, (void *) mat_ptr, val_type
        );
    }
}

void krnl_grad_func_omp(
    const int n0, const int ld0, const void *c0,
    const int n1, const int ld1, const void *c1,
    krnl_grad_func krnl_grad, const void *param, const int ldm, 
    const int require_krnl, void *k_mat, 
    const int require_grad, void *dl_mat, 
    const int val_type, const int n_thread
)
{
    size_t val_bytes = (val_type == VAL_TYPE_DOUBLE) ? sizeof(double) : sizeof(float);
    int n_thread_ = (n_thread <= 0) ? omp_get_max_threads() : n_thread;
    if (omp_get_num_threads() > 1) n_thread_ = 1;
    int nt0, nt1;
    find_2d_decomp(n0, n1, n_thread_, &nt0, &nt1);
    #pragma omp parallel num_threads(n_thread_)
    {
        int tid = omp_get_thread_num();
        int tid0 = tid % nt0, tid1 = tid / nt0;
        int srow = 0, nrow = 0, scol = 0, ncol = 0;
        calc_block_spos_len(n0, nt0, tid0, &srow, &nrow);
        calc_block_spos_len(n1, nt1, tid1, &scol, &ncol);
        const char *c0_ptr = ((char *) c0) + srow * val_bytes;
        const char *c1_ptr = ((char *) c1) + scol * val_bytes;
        size_t mat_displs = (srow * val_bytes) + (scol * ldm * val_bytes);
        char *k_mat_ptr = NULL, *dl_mat_ptr = NULL;
        if (require_krnl == 1) k_mat_ptr  = ((char *) k_mat ) + mat_displs;
        if (require_grad == 1) dl_mat_ptr = ((char *) dl_mat) + mat_displs;
        krnl_grad(
            nrow, ld0, (const void *) c0_ptr, 
            ncol, ld1, (const void *) c1_ptr, 
            param, ldm, val_type, 
            require_krnl, (void *) k_mat_ptr, 
            require_grad, (void *) dl_mat_ptr
        );
    }
}

template<typename VT>
static void krnl_matmul_omp(
    const int n0, const int ld0, const VT *c0,
    const int n1, const int ld1, const VT *c1,
    krnl_func krnl, const VT *param, const int val_type, 
    const int nvec, const VT *B, const int ldB, 
    VT *C, const int ldC, const int n_thread
)
{
    size_t val_bytes = (val_type == VAL_TYPE_DOUBLE) ? sizeof(double) : sizeof(float);
    int n_thread_ = (n_thread <= 0) ? omp_get_max_threads() : n_thread;
    if (omp_get_num_threads() > 1) n_thread_ = 1;

    int k = 1;
    while (n_thread_ * k * 512 < n0) k++;
    int num_blk0 = n_thread_ * k;
    int blk_size0 = (n0 + num_blk0 - 1) / num_blk0;
    int blk_size1 = 512;

    VT v_one = 1.0;
    VT *k_buf = (VT *) malloc(blk_size0 * blk_size1 * val_bytes * n_thread_);
    ASSERT_PRINTF(k_buf != NULL, "Failed to allocate memory for %s\n", __FUNCTION__);

    #pragma omp parallel num_threads(n_thread_)
    {
        int tid = omp_get_thread_num();
        VT *local_k = k_buf  + tid * blk_size0 * blk_size1;

        #pragma omp for schedule(dynamic)
        for (int srow = 0; srow < n0; srow += blk_size0)
        {
            int nrow = (srow + blk_size0 <= n0) ? blk_size0 : n0 - srow;
            VT *C_ptr = C + srow;
            
            for (int scol = 0; scol < n1; scol += blk_size1)
            {
                int ncol = (scol + blk_size1 <= n1) ? blk_size1 : n1 - scol;
                const VT *B_ptr  = B  + scol;
                const VT *c0_ptr = c0 + srow;
                const VT *c1_ptr = c1 + scol;
                krnl(
                    nrow, ld0, (const void *) c0_ptr, 
                    ncol, ld1, (const void *) c1_ptr, 
                    param, nrow, (void *) local_k, val_type
                );

                VT beta = (scol == 0) ? 0.0 : 1.0;
                xgemm_(
                    notrans, notrans, &nrow, &nvec, &ncol,
                    &v_one, local_k, &nrow, B_ptr, &ldB,
                    &beta, C_ptr, &ldC
                );
            }  // End of scol loop
        }  // End of srow loop
    }  // End of "#pragma omp parallel"

    free(k_buf);
}

void krnl_matmul_omp(
    const int n0, const int ld0, const void *c0,
    const int n1, const int ld1, const void *c1,
    krnl_func krnl, const void *param, const int val_type, 
    const void *B, const int ldB, const int nvec, 
    void *C, const int ldC, const int n_thread
)
{
    if (val_type == VAL_TYPE_DOUBLE)
    {
        krnl_matmul_omp<double>(
            n0, ld0, (const double *) c0, n1, ld1, (const double *) c1,
            krnl, (const double *) param, val_type, 
            nvec,(const double *) B, ldB, (double *) C, ldC, n_thread
        );
    }
    if (val_type == VAL_TYPE_FLOAT)
    {
        krnl_matmul_omp<float>(
            n0, ld0, (const float *)  c0, n1, ld1, (const float *)  c1,
            krnl, (const float *)  param, val_type, 
            nvec, (const float *)  B, ldB, (float *)  C, ldC, n_thread
        );
    }
}

template<typename VT>
static void krnl_grad_matmul_omp(
    const int n0, const int ld0, const VT *c0,
    const int n1, const int ld1, const VT *c1,
    krnl_grad_func krnl_grad, const VT *param, const int val_type, 
    const VT *B, const int ldB, const int nvec, 
    VT *krnl_C, VT *grad_C, const int ldC, const int n_thread
)
{
    size_t val_bytes = (val_type == VAL_TYPE_DOUBLE) ? sizeof(double) : sizeof(float);
    int n_thread_ = (n_thread <= 0) ? omp_get_max_threads() : n_thread;
    if (omp_get_num_threads() > 1) n_thread_ = 1;

    const int require_krnl = (krnl_C != NULL) ? 1 : 0;
    const int require_grad = (grad_C != NULL) ? 1 : 0;

    int k = 1;
    while (n_thread_ * k * 512 < n0) k++;
    int num_blk0 = n_thread_ * k;
    int blk_size0 = (n0 + num_blk0 - 1) / num_blk0;
    int blk_size1 = 512;

    VT v_one = 1.0;
    VT *k_buf  = (VT *) malloc(blk_size0 * blk_size1 * val_bytes * n_thread_);
    VT *dl_buf = (VT *) malloc(blk_size0 * blk_size1 * val_bytes * n_thread_);
    ASSERT_PRINTF(k_buf != NULL && dl_buf != NULL, "Failed to allocate memory for %s\n", __FUNCTION__);

    #pragma omp parallel num_threads(n_thread_)
    {
        int tid = omp_get_thread_num();
        VT *local_k  = k_buf  + tid * blk_size0 * blk_size1;
        VT *local_dl = dl_buf + tid * blk_size0 * blk_size1;

        #pragma omp for schedule(dynamic)
        for (int srow = 0; srow < n0; srow += blk_size0)
        {
            int nrow = (srow + blk_size0 <= n0) ? blk_size0 : n0 - srow;
            VT *krnl_C_ptr = krnl_C + srow;
            VT *grad_C_ptr = grad_C + srow;
            
            for (int scol = 0; scol < n1; scol += blk_size1)
            {
                int ncol = (scol + blk_size1 <= n1) ? blk_size1 : n1 - scol;
                const VT *B_ptr  = B  + scol;
                const VT *c0_ptr = c0 + srow;
                const VT *c1_ptr = c1 + scol;
                krnl_grad(
                    nrow, ld0, (const void *) c0_ptr, 
                    ncol, ld1, (const void *) c1_ptr, 
                    param, nrow, val_type, 
                    require_krnl, (void *) local_k, 
                    require_grad, (void *) local_dl
                );

                VT beta = (scol == 0) ? 0.0 : 1.0;
                if (require_krnl)
                {
                    xgemm_(
                        notrans, notrans, &nrow, &nvec, &ncol,
                        &v_one, local_k, &nrow, B_ptr, &ldB,
                        &beta, krnl_C_ptr, &ldC
                    );
                }
                if (require_grad)
                {
                    xgemm_(
                        notrans, notrans, &nrow, &nvec, &ncol,
                        &v_one, local_dl, &nrow, B_ptr, &ldB,
                        &beta, grad_C_ptr, &ldC
                    );
                }
            }  // End of scol loop
        }  // End of srow loop
    }  // End of "#pragma omp parallel"

    free(k_buf);
    free(dl_buf);
}

void krnl_grad_matmul_omp(
    const int n0, const int ld0, const void *c0,
    const int n1, const int ld1, const void *c1,
    krnl_grad_func krnl_grad, const void *param, const int val_type, 
    const void *B, const int ldB, const int nvec, 
    void *krnl_C, void *grad_C, const int ldC, const int n_thread
)
{
    if (val_type == VAL_TYPE_DOUBLE)
    {
        krnl_grad_matmul_omp<double>(
            n0, ld0, (const double *) c0, n1, ld1, (const double *) c1,
            krnl_grad, (const double *) param, val_type,
            (const double *) B, ldB, nvec,
            (double *) krnl_C, (double *) grad_C, ldC, n_thread
        );
    }
    if (val_type == VAL_TYPE_FLOAT)
    {
        krnl_grad_matmul_omp<float>(
            n0, ld0, (const float *)  c0, n1, ld1, (const float *)  c1,
            krnl_grad, (const float *)  param, val_type,
            (const float *)  B, ldB, nvec,
            (float *)  krnl_C, (float *)  grad_C, ldC, n_thread
        );
    }
    
}

