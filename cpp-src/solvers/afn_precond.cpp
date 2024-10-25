#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <limits>
#include <omp.h>

#include "afn_precond.h"
#include "../common.h"
#include "../cpu_linalg.hpp"
#include "../utils.h"
#include "../dense_kernel_matrix.h"
#include "fsai_precond.h"

// Select k points from X using Farthest Point Sampling (FPS)
// Input parameters:
//   npt    : Number of points in coord
//   pt_dim : Dimension of each point
//   coord  : Matrix, size ldc * pt_dim, col-major, each row is a point coordinate
//   k      : Number of points to select, <= npt
// Output parameter:
//   idx : Vector, size min(k, npt), indices of selected points
template<typename VT>
static void fps(const int npt, const int pt_dim, const VT *coord, const int ldc, const int k, int *idx)
{
    VT *workbuf = (VT *) malloc(sizeof(VT) * (2 * npt + pt_dim));
    VT *center = workbuf;
    VT *tmp_d  = center + pt_dim;
    VT *min_d  = tmp_d  + npt;
    ASSERT_PRINTF(workbuf != NULL, "Failed to allocate work arrays for %s\n", __FUNCTION__);

    memset(center, 0, sizeof(VT) * pt_dim);
    for (int i = 0; i < pt_dim; i++)
    {
        const VT *coord_i = coord + i * ldc;
        #pragma omp simd
        for (int j = 0; j < npt; j++) center[i] += coord_i[j];
        center[i] /= (VT) npt;
    }

    int n_thread = omp_get_max_threads();
    int *min_d_idx = (int *) malloc(sizeof(int) * n_thread);
    VT  *min_d_val = (VT *)  malloc(sizeof(VT)  * n_thread);
    int min_d_idx0;
    #pragma omp parallel num_threads(n_thread) 
    {
        // (1) Calculate the distance of all points to the center
        #pragma omp for schedule(static)
        for (int j = 0; j < npt; j++) tmp_d[j] = 0.0;
        for (int d = 0; d < pt_dim; d++)
        {
            VT center_d = center[d];
            const VT *coord_d = coord + d * ldc;
            #pragma omp for schedule(static)
            for (int j = 0; j < npt; j++)
            {
                VT diff = coord_d[j] - center_d;
                tmp_d[j] += diff * diff;
            }
        }

        // (2) Each thread find its local minimum in tmp_d
        #pragma omp barrier
        int tid = omp_get_thread_num();
        int tmp_idx = 0;
        VT tmp_val = tmp_d[0];
        #pragma omp for schedule(static)
        for (int j = 0; j < npt; j++)
            if (tmp_d[j] < tmp_val) { tmp_val = tmp_d[j]; tmp_idx = j; }
        min_d_idx[tid] = tmp_idx;
        min_d_val[tid] = tmp_val;
    }
    // (3) Find the global minimum distance to center
    int tmp_idx2 = 0;
    VT tmp_val2 = min_d_val[0];
    for (int t = 1; t < n_thread; t++)
        if (min_d_val[t] < tmp_val2) { tmp_val2 = min_d_val[t]; tmp_idx2 = t; }
    idx[0] = min_d_idx[tmp_idx2];
    min_d_idx0 = idx[0];

    // Find the rest k - 1 points
    for (int i = 1; i < k; i++)
    {
        #pragma omp parallel num_threads(n_thread) 
        {
            // (1) Calculate the distance of all points to the last selected point and update min_d
            #pragma omp for schedule(static)
            for (int j = 0; j < npt; j++) tmp_d[j] = 0.0;
            for (int d = 0; d < pt_dim; d++)
            {
                VT last_d = coord[d * ldc + min_d_idx0];
                const VT *coord_d = coord + d * ldc;
                #pragma omp for schedule(static)
                for (int j = 0; j < npt; j++)
                {
                    VT diff = coord_d[j] - last_d;
                    tmp_d[j] += diff * diff;
                }
            }
            if (i == 1)
            {
                #pragma omp for schedule(static)
                for (int j = 0; j < npt; j++) min_d[j] = tmp_d[j];
            } else {
                #pragma omp for schedule(static)
                for (int j = 0; j < npt; j++) 
                    min_d[j] = (tmp_d[j] < min_d[j]) ? tmp_d[j] : min_d[j];
            }

            // (2) Each thread find its local maximum in min_d
            #pragma omp barrier
            int tid = omp_get_thread_num();
            int tmp_idx = 0;
            VT tmp_val = 0;
            #pragma omp for schedule(static)
            for (int j = 0; j < npt; j++)
                if (min_d[j] > tmp_val) { tmp_val = min_d[j]; tmp_idx = j; }
            min_d_idx[tid] = tmp_idx;
            min_d_val[tid] = tmp_val;
        }
        // (3) Find the global maximum distance to the last selected point
        tmp_idx2 = 0;
        tmp_val2 = min_d_val[0];
        for (int t = 1; t < n_thread; t++)
            if (min_d_val[t] > tmp_val2) { tmp_val2 = min_d_val[t]; tmp_idx2 = t; }
        idx[i] = min_d_idx[tmp_idx2];
        min_d_idx0 = idx[i];
    }

    free(min_d_idx);
    free(min_d_val);
    free(workbuf);
}

// FPS permutation, the only difference with fps() is that the length of idx is now npt, 
// and the values after the first k elements are filled from 0 to npt-1 except the first k values
template<typename VT>
static void fps_perm(const int npt, const int pt_dim, const VT *coord, const int ldc, const int k, int *idx)
{
    char *flag = (char *) malloc(sizeof(char) * npt);
    ASSERT_PRINTF(flag != NULL, "Failed to allocate work buffer for %s\n", __FUNCTION__);
    memset(flag, 0, sizeof(char) * npt);
    fps<VT>(npt, pt_dim, coord, ldc, k, idx);
    for (int i = 0; i < k; i++) flag[idx[i]] = 1;
    int idx1 = k;
    for (int i = 0; i < npt; i++)
        if (flag[i] == 0) idx[idx1++] = i;
    free(flag);
}

// Fisher-Yates shuffle for random permutation
// Input parameter:
//   m : Number of elements to shuffle
// Output parameter:
//   rand_idx : Size m, randomly shuffled indices ranging from 0 to m-1
static void randperm(const int m, int *rand_idx)
{
    for (int i = 0; i < m; i++) rand_idx[i] = i;
    for (int i = 0; i < m / 2; i++)
    {
        int j = (rand() % (m - i)) + i;
        int tmp = rand_idx[i];
        rand_idx[i] = rand_idx[j];
        rand_idx[j] = tmp;
    }
}

// Sample and scale a subset of points for estimating the rank of a kernel matrix
// Input and output parameters are the same as afn_rank_est()
template<typename VT>
static int afn_rank_est_scaled(
    const int npt, const int pt_dim, const VT *coord, const int ldc, 
    const int krnl_id, const VT *param, const int npt_s, const int n_rep
)
{
    int val_type  = (sizeof(VT) == sizeof(double)) ? VAL_TYPE_DOUBLE : VAL_TYPE_FLOAT;
    VT diag_shift = param[3];

    int sample_r = 0, r = 0;
    VT v_base = (VT) npt_s / (VT) npt;
    VT v_exp = (VT) 1.0 / (VT) pt_dim;
    VT scale_factor = std::pow(v_base, v_exp);

    VT  *workbuf  = (VT *)  malloc(sizeof(VT) * (npt_s * (pt_dim + 4 * npt_s)));
    int *fps_perm = (int *) malloc(sizeof(int) * npt_s);
    int *rand_idx = (int *) malloc(sizeof(int) * npt);
    ASSERT_PRINTF(
        workbuf != NULL && fps_perm != NULL && rand_idx != NULL,
        "Failed to allocate work arrays for %s\n", __FUNCTION__
    );
    VT *Xs   = workbuf;
    VT *Ks   = Xs  + npt_s * pt_dim;
    VT *K11  = Ks  + npt_s * npt_s;
    VT *K1   = K11 + npt_s * npt_s;
    VT *Knys = K1  + npt_s * npt_s;

    for (int i_rep = 0; i_rep < n_rep; i_rep++)
    {
        // Randomly sample and scale npt_s points
        randperm(npt, rand_idx);
        gather_matrix_cols(sizeof(VT), pt_dim, npt_s, rand_idx, (const void *) coord, ldc, (void *) Xs, npt_s);
        for (int i = 0; i < npt_s * pt_dim; i++) Xs[i] *= scale_factor;

        // Reorder points using FPS for Nystrom (use Ks as a temporary buffer)
        fps<VT>(npt_s, pt_dim, Xs, npt_s, npt_s, fps_perm);
        gather_matrix_cols(sizeof(VT), pt_dim, npt_s, fps_perm, (const void *) Xs, npt_s, (void *) Ks, npt_s);
        memcpy(Xs, Ks, sizeof(VT) * npt_s * pt_dim);

        // Compute Ks and remove its diagonal shift
        dense_krnl_mat_p dk_Ks = NULL;
        dense_krnl_mat_init(npt_s, npt_s, Xs, npt_s, npt_s, Xs, (const void *) param, NULL, krnl_id, val_type, &dk_Ks);
        dense_krnl_mat_grad_eval(dk_Ks, (void *) Ks, NULL, NULL, NULL);
        dense_krnl_mat_free(&dk_Ks);
        
        // Shift the Ks matrix to make Nystrom stable
        VT Ks_fnorm = 0;
        for (int i = 0; i < npt_s * npt_s; i++) Ks_fnorm += Ks[i] * Ks[i];
        Ks_fnorm = std::sqrt(Ks_fnorm);
        VT nu = std::sqrt((VT) npt_s) * Ks_fnorm * std::numeric_limits<VT>::epsilon();
        for (int i = 0; i < npt_s; i++) Ks[i * npt_s + i] += nu - diag_shift;

        // Binary search to find the minimal rank
        int rs = 1, re = npt_s, rc;
        while (rs < re)
        {
            // Ks is reordered by FPS, use the first rc rows & columns as Nystrom basis
            // copy_matrix works for row-major matrix, swap row & column parameters
            // K11 = Ks(1 : rc, 1 : rc);
            // K1  = Ks(1 : rc, :);
            rc = (rs + re) / 2;
            copy_matrix(sizeof(VT), rc,    rc, Ks, npt_s, K11, rc, 0);
            copy_matrix(sizeof(VT), npt_s, rc, Ks, npt_s, K1,  rc, 0);
            
            // Knys = K1' * (K11 \ K1);
            int info = 0;
            VT v_zero = 0, v_one = 1;
            xposv_(lower, &rc, &npt_s, K11, &rc, K1, &rc, &info);
            ASSERT_PRINTF(info == 0, "xPOSV returned %d\n", info);
            xgemm_(
                notrans, notrans, &npt_s, &npt_s, &rc,
                &v_one, Ks, &npt_s, K1, &rc,
                &v_zero, Knys, &npt_s
            );

            VT err_fnorm = 0, relerr = 0;
            for (int i = 0; i < npt_s * npt_s; i++)
            {
                VT diff = Ks[i] - Knys[i];
                err_fnorm += diff * diff;
            }
            err_fnorm = std::sqrt(err_fnorm);
            relerr = err_fnorm / Ks_fnorm;
            if (relerr < 0.1) re = rc; 
            else rs = rc + 1;
        }  // End of while (rs < re)
        sample_r = (rs > sample_r) ? rs : sample_r;
    }  // End of i_rep loop
    
    r = std::ceil((VT) sample_r * (VT) npt / (VT) npt_s);
    free(workbuf);
    free(fps_perm);
    free(rand_idx);
    return r;
}

// Estimate the rank of a kernel matrix for Nystrom preconditioner
// Input and output parameters are the same as afn_rank_est()
template<typename VT>
static int afn_rank_est_nys(
    const int npt, const int pt_dim, const VT *coord, const int ldc, const int krnl_id, 
    const VT *param, const VT *dnoise, const int npt_s, const int n_rep, const int max_k
)
{
    int val_type  = (sizeof(VT) == sizeof(double)) ? VAL_TYPE_DOUBLE : VAL_TYPE_FLOAT;
    VT diag_shift = param[3];

    int sample_r = 0, r = 0, max_k_ = max_k;
    if (max_k_ < npt_s) max_k_ = npt_s;

    VT *workbuf = (VT *) malloc(sizeof(VT) * (max_k_ * pt_dim + max_k_ * (max_k_ * 2 + 1)));
    int *rand_idx = (int *) malloc(sizeof(int) * npt);
    ASSERT_PRINTF(workbuf != NULL && rand_idx != NULL, "Failed to allocate work arrays for %s\n", __FUNCTION__);
    VT *Xs   = workbuf;
    VT *Ks   = Xs + max_k_ * pt_dim;
    VT *ev   = Ks + max_k_ * max_k_;
    VT *work = ev + max_k_;

    int lwork = max_k_ * max_k_, info = 0;
    VT scale_factor = 1.0;
    int npt_s_ = max_k_;  // npt_s_ is the number of points to sample
    if (max_k_ > npt_s)
    {
        VT v_base = (VT) npt_s / (VT) max_k_;
        VT v_exp = (VT) 1.0 / (VT) pt_dim;
        scale_factor = std::pow(v_base, v_exp);
        npt_s_ = npt_s;
    }
    VT *dn1 = (VT *) malloc(sizeof(VT) * npt_s_);

    for (int i_rep = 0; i_rep < n_rep; i_rep++)
    {
        // Randomly sample and scale npt_s_ points
        randperm(npt, rand_idx);
        gather_matrix_cols(sizeof(VT), pt_dim, npt_s_, rand_idx, (const void *) coord, ldc, (void *) Xs, npt_s_);
        for (int i = 0; i < npt_s_ * pt_dim; i++) Xs[i] *= scale_factor;
        for (int i = 0; i < npt_s_; i++) dn1[i] = dnoise[rand_idx[i]];

        // Ks = kernel(Xs, Xs);
        // Ks already has the diagonal shift
        // ev = eig(Ks);
        dense_krnl_mat_p dk_Ks = NULL;
        dense_krnl_mat_init(npt_s_, npt_s_, Xs, npt_s_, npt_s_, Xs, (const void *) param, dn1, krnl_id, val_type, &dk_Ks);
        dense_krnl_mat_grad_eval(dk_Ks, (void *) Ks, NULL, NULL, NULL);
        dense_krnl_mat_free(&dk_Ks);
        xsyev_(nocalc, lower, &npt_s_, Ks, &npt_s_, ev, work, &lwork, &info);
        ASSERT_PRINTF(info == 0, "xSYEV returned %d\n", info);

        // sample_r = max(sample_r, sum(ev > 1.1 * mu));
        int rc = 0;
        VT threshold = 1.1 * diag_shift;
        for (int i = 0; i < npt_s_; i++) if (ev[i] > threshold) rc++;
        sample_r = (rc > sample_r) ? rc : sample_r;
    }  // End of i_rep loop

    r = std::ceil((VT) sample_r * (VT) max_k_ / (VT) npt_s);
    free(rand_idx);
    free(workbuf);
    free(dn1);
    return r;
}

// Estimate the rank of a kernel matrix for AFN preconditioner
// Input parameters:
//   npt     : Number of points in coord
//   pt_dim  : Dimension of each point
//   coord   : Matrix, size ldc * pt_dim, col-major, each row is a point coordinate
//   ldc     : Leading dimension of coord, >= npt
//   krnl_id : Kernel ID, see kernels/kernels.h
//   param   : Pointer to kernel function parameter array, [dim, l, f, s]
//   npt_s   : Number of points to sample
//   n_rep   : Number of times to repeat the estimation
//   max_k   : Max global low-rank approximation rank (K11's size)
// Output parameter:
//   <return> : Estimated rank of the kernel matrix
template<typename VT>
static int afn_rank_est(
    const int npt, const int pt_dim, const VT *coord, const int ldc, const int krnl_id, 
    const VT *param, const VT *dnoise, const int npt_s, const int n_rep, const int max_k
)
{
    int r_scaled = afn_rank_est_scaled<VT>(npt, pt_dim, coord, ldc, krnl_id, param, npt_s, n_rep);
    if (r_scaled > max_k) return r_scaled;
    else {
        // Estimated rank is small, will use Nystrom instead of AFN, use
        // original points without scaling to better estimate the rank
        int r_nys = afn_rank_est_nys<VT>(npt, pt_dim, coord, ldc, krnl_id, param, dnoise, npt_s, n_rep, max_k);
        return r_nys;
    }
}

template<typename VT>
static void afn_precond_build(
    const int val_type, const int krnl_id, const VT *param, const VT *dnoise_, 
    const int npt, const int pt_dim, const VT *coord, const int ldc, 
    const int npt_s, const int glr_rank, const int fsai_npt, 
    octree_p octree, const int need_grad, afn_precond_p *ap
)
{
    afn_precond_p ap_ = (afn_precond_p) malloc(sizeof(afn_precond_s));
    memset(ap_, 0, sizeof(afn_precond_s));

    size_t VT_bytes = (sizeof(VT) == sizeof(double)) ? sizeof(double) : sizeof(float);
    int n_grad = need_grad ? 3 : 0;

    ap_->val_type = val_type;
    ap_->n = npt;

    VT *dnoise = (VT *) malloc(sizeof(VT) * npt);
    memset(dnoise, 0, sizeof(VT) * npt);
    if (dnoise_ != NULL) memcpy(dnoise, dnoise_, sizeof(VT) * npt);

    // 1. Estimate kernel matrix rank and determine if we should fall back to Nystrom only
    int est_rank = 0;
    if (npt_s > 0)
    {
        est_rank = afn_rank_est<VT>(npt, pt_dim, coord, ldc, krnl_id, param, dnoise, npt_s, 1, glr_rank);
    } else {
        // Always use AFN, do not fall back to Nystrom
        est_rank = -npt_s;
    }
    ap_->est_rank = est_rank;
    if ((est_rank <= glr_rank) || (fsai_npt <= 0))
    {
        if ((fsai_npt <= 0) && (est_rank > glr_rank)) est_rank = glr_rank;
        int *perm = (int *) malloc(sizeof(int) * npt);
        ASSERT_PRINTF(perm != NULL, "Failed to allocate work buffer for %s\n", __FUNCTION__);
        fps_perm<VT>(npt, pt_dim, coord, ldc, est_rank, perm);
        nys_precond_build(
            val_type, krnl_id, (const void *) param, dnoise, 
            npt, pt_dim, coord, ldc,
            perm, est_rank, need_grad, &ap_->np
        );
        ap_->is_nys   = 1;
        ap_->n1       = est_rank;
        ap_->n2       = 0;
        ap_->fsai_npt = 0;
        ap_->logdet   = malloc(sizeof(VT));
        memcpy(ap_->logdet, ap_->np->logdet, sizeof(VT));
        if (need_grad)
        {
            ap_->gt = (VT *) malloc(sizeof(VT) * n_grad);
            nys_precond_grad_trace(ap_->np);
            memcpy(ap_->gt, ap_->np->gt, sizeof(VT) * n_grad);
        }
        free(perm);
        *ap = ap_;
        return;
    } else {
        ap_->is_nys   = 0;
        ap_->n1       = glr_rank;
        ap_->n2       = npt - glr_rank;
        ap_->fsai_npt = fsai_npt;
    }

    // 2. Use FPS to select n1 points, swap them to the front
    int n = npt, n1 = glr_rank, n2 = npt - n1;
    int *perm = (int *) malloc(sizeof(int)  * n);
    VT *coord_perm = (VT *) malloc(sizeof(VT) * npt * pt_dim);
    ASSERT_PRINTF(
        perm != NULL && coord_perm != NULL,
        "Failed to allocate AFN preconditioner FPS buffers\n"
    );
    fps_perm<VT>(npt, pt_dim, coord, ldc, n1, perm);
    // gather_matrix_cols works on row-major matrices, swap row & column parameters
    gather_matrix_cols(VT_bytes, pt_dim, npt, perm, (const void *) coord, ldc, (void *) coord_perm, npt);
    ap_->perm = perm;

    // 3. Find fsai_npt nearest neighbors for each point in the last n2 points
    int *nn_idx    = (int *) malloc(sizeof(int) * n2 * fsai_npt);
    int *nn_displs = (int *) malloc(sizeof(int) * (n2 + 1));
    VT *coord_1 = coord_perm + n1;
    if (octree == NULL)
    {
        fsai_exact_knn(
            ap_->val_type, fsai_npt, n2, pt_dim, 
            (const void *) coord_1, n, nn_idx, nn_displs + 1
        );
    } else {
        int *coord0_idx = perm + n1;
        fsai_octree_fast_knn(
            ap_->val_type, fsai_npt, n2, pt_dim, 
            coord_perm + n1, n, coord0_idx, octree, nn_idx, nn_displs + 1
        );
    }
    nn_displs[0] = 0;
    for (int i = 1; i <= n2; i++) nn_displs[i] += nn_displs[i - 1];

    // 4. Compute K11 and K12
    int ldK11 = n1, ldK12 = n1;
    dense_krnl_mat_p dK_11 = NULL, dK_12 = NULL;
    VT *coord_n1 = coord_perm;
    VT *coord_n2 = coord_perm + n1;
    VT *K11 = (VT *) malloc(sizeof(VT) * n1 * n1);
    VT *K12 = (VT *) malloc(sizeof(VT) * n1 * n2);
    VT *dK11 = NULL, *dK11_l = NULL, *dK11_f = NULL, *dK11_s = NULL;
    VT *dK12 = NULL, *dK12_l = NULL, *dK12_f = NULL, *dK12_s = NULL;
    VT *dnoise_perm = (VT *) malloc(sizeof(VT) * n);
    ASSERT_PRINTF(
        K11 != NULL && K12 != NULL && dnoise_perm != NULL,
        "Failed to allocate work arrays for %s\n", __FUNCTION__
    );
    for (int i = 0; i < n; i++) dnoise_perm[i] = dnoise[perm[i]];
    VT *dnoise1 = dnoise_perm;
    VT *dnoise2 = dnoise_perm + n1;
    dense_krnl_mat_init(
        n1, n, coord_n1, n1, n, coord_n1, 
        (const void*) param, dnoise1, krnl_id, val_type, &dK_11
    );
    dense_krnl_mat_init(
        n1, n, coord_n1, n2, n, coord_n2, 
        (const void*) param, NULL, krnl_id, val_type, &dK_12
    );
    if (need_grad)
    {
        dK11 = (VT *) malloc(sizeof(VT) * n1 * n1 * n_grad);
        dK12 = (VT *) malloc(sizeof(VT) * n1 * n2 * n_grad);
        ASSERT_PRINTF(
            dK11 != NULL && dK12 != NULL,
            "Failed to allocate AFN preconditioner dK11/dK12 arrays\n"
        );
        dK11_l = dK11;
        dK11_f = dK11 + n1 * n1;
        dK11_s = dK11 + n1 * n1 * 2;
        dK12_l = dK12;
        dK12_f = dK12 + n1 * n2;
        dK12_s = dK12 + n1 * n2 * 2;
    }
    dense_krnl_mat_grad_eval(dK_11, K11, dK11_l, dK11_f, dK11_s);
    dense_krnl_mat_grad_eval(dK_12, K12, dK12_l, dK12_f, dK12_s);
    dense_krnl_mat_free(&dK_11);
    dense_krnl_mat_free(&dK_12);

    // 5. Compute L11 = chol(K11, 'lower') and solve L11 * V12 = K12
    int info;
    xpotrf_(lower, &n1, K11, &ldK11, &info);
    ASSERT_PRINTF(info == 0, "LAPACK xPOTRF return %d\n", info);
    for (int j = 1; j < n1; j++)
        memset(K11 + j * ldK11, 0, sizeof(VT) * j);
    VT *V12 = (VT *) malloc(sizeof(VT) * n1 * n2);
    ASSERT_PRINTF(V12 != NULL, "Failed to allocate AFN preconditioner V12 work buffer\n");
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n1 * n2; i++) V12[i] = K12[i];
    VT v_one = 1.0, v_zero = 0.0;
    xtrsm_(left, lower, notrans, nonunit, &n1, &n2, &v_one, K11, &ldK11, V12, &ldK12);
    ap_->L11 = K11;
    ap_->K12 = K12;
    VT *L11 = K11;

    // 6. Gradient matrices for AFN trace computation
    VT *GdKG = NULL, *dL11 = NULL, *tmpM = NULL;
    VT *GdK12 = NULL, *GdV12 = NULL;
    if (need_grad)
    {
        GdKG  = (VT *) malloc(sizeof(VT) * n1 * n1 * n_grad);
        dL11  = (VT *) malloc(sizeof(VT) * n1 * n1 * n_grad);
        GdK12 = (VT *) malloc(sizeof(VT) * n1 * n2 * n_grad);
        GdV12 = (VT *) malloc(sizeof(VT) * n1 * n2 * n_grad);
        tmpM  = (VT *) malloc(sizeof(VT) * n1 * n1);
        ASSERT_PRINTF(
            GdKG != NULL && dL11 != NULL && tmpM != NULL && GdK12 != NULL && GdV12 != NULL,
            "Failed to allocate AFN preconditioner work arrays\n"
        );
    }
    for (int i_grad = 0; i_grad < n_grad; i_grad++)
    {
        VT *dK11_i = dK11 + i_grad * n1 * n1;
        VT *GdKG_i = GdKG + i_grad * n1 * n1;

        // (1) GdKG{i} = inv(L11) * dKi * inv(L11)';
        // Solve 1: L11 * tmpM = dKi
        // Solve 2: GdKG{i} * L11^T = tmpM;
        memcpy(tmpM, dK11_i, sizeof(VT) * n1 * n1);
        xtrsm_(left,  lower, notrans, nonunit, &n1, &n1, &v_one, L11, &ldK11, tmpM, &ldK11);
        xtrsm_(right, lower, trans,   nonunit, &n1, &n1, &v_one, L11, &ldK11, tmpM, &ldK11);
        memcpy(GdKG_i, tmpM, sizeof(VT) * n1 * n1);

        // (2) dL11{i} = L11 * PHI(GdKG{i});
        // PHI(K) = tril(K, -1) + diag(diag(K) * 0.5);
        VT *dL11_i = dL11 + i_grad * n1 * n1;
        for (int i = 0; i < n1; i++)
        {
            for (int j = 0; j < i; j++) tmpM[i * ldK11 + j] = 0.0;
            tmpM[i * ldK11 + i] *= 0.5;
        }
        xgemm_(
            notrans, notrans, &n1, &n1, &n1,
            &v_one, L11, &ldK11, tmpM, &ldK11,
            &v_zero, dL11_i, &ldK11
        );

        // (3) GdK12{i} = L11 \ dK12{i};
        // (4) GdV12{i} = GdKG{i} * V12;
        VT *dK12_i  = dK12  + i_grad * n1 * n2;
        VT *GdK12_i = GdK12 + i_grad * n1 * n2;
        VT *GdV12_i = GdV12 + i_grad * n1 * n2;
        memcpy(GdK12_i, dK12_i, sizeof(VT) * n1 * n2);
        xtrsm_(left, lower, notrans, nonunit, &n1, &n2, &v_one, L11, &ldK11, GdK12_i, &ldK12); 
        xgemm_(
            notrans, notrans, &n1, &n2, &n1,
            &v_one, GdKG_i, &ldK11, V12, &ldK12,
            &v_zero, GdV12_i, &ldK12
        );
    }  // End of i_grad loop
    free(dK11);
    free(GdKG);
    free(tmpM);
    ap_->dL11 = dL11;
    ap_->dK12 = dK12;

    // 7. FSAI for S = K22 - V12^T * V12
    fsai_precond_build(
        val_type, krnl_id, param, dnoise2, 
        n2, pt_dim, coord_n2, n,
        fsai_npt, nn_idx, nn_displs, 
        n1, V12, need_grad, 
        GdK12, GdV12, &ap_->fp
    );
    free(nn_idx);
    free(nn_displs);
    free(V12);
    free(GdK12);
    free(GdV12);

    // 8. Compute logdet = 2 * (sum(log(diag(ap.L11))) + sum(log(1./diag(ap.G))))
    VT logdet = 0.0;
    int *G_rowptr = ap_->fp->G->rowptr;
    VT  *G_val    = (VT *) ap_->fp->G->val;
    for (int i = 0; i < n1; i++) logdet += std::log(K11[i * ldK11 + i]);
    for (int i = 0; i < n2; i++)
    {
        int G_ii_idx = G_rowptr[i + 1] - 1;
        logdet += std::log(1.0 / G_val[G_ii_idx]);
    }
    logdet *= 2.0;
    ap_->logdet = malloc(sizeof(VT));
    memcpy(ap_->logdet, &logdet, sizeof(VT));

    if (need_grad) afn_precond_grad_trace(ap_);

    *ap = ap_;
}

// Build an AFN preconditioner for f^2 * (K(X, X, l) + s * I)
void afn_precond_build(
    const int val_type, const int krnl_id, const void *param, const void *dnoise, 
    const int npt, const int pt_dim, const void *coord, const int ldc, 
    const int npt_s, const int glr_rank, const int fsai_npt, 
    octree_p octree, const int need_grad, afn_precond_p *ap
)
{
    if (val_type == VAL_TYPE_DOUBLE)
    {
        afn_precond_build<double>(
            val_type, krnl_id, (const double *) param, (const double *) dnoise, 
            npt, pt_dim, (const double *) coord, ldc, 
            npt_s, glr_rank, fsai_npt, octree, need_grad, ap
        );
    }
    if (val_type == VAL_TYPE_FLOAT)
    {
        afn_precond_build<float>(
            val_type, krnl_id, (const float *)  param, (const float *)  dnoise, 
            npt, pt_dim, (const float *)  coord, ldc, 
            npt_s, glr_rank, fsai_npt, octree, need_grad, ap
        );
    }
}

// Free an initialized afn_precond struct
void afn_precond_free(afn_precond_p *ap)
{
    afn_precond_p ap_ = *ap;
    if (ap_ == NULL) return;
    free(ap_->perm);
    free(ap_->L11);
    free(ap_->K12);
    free(ap_->dL11);
    free(ap_->dK12);
    free(ap_->logdet);
    free(ap_->gt);
    nys_precond_free(&ap_->np);
    fsai_precond_free(&ap_->fp);
    free(ap_);
    *ap = NULL;
}

template<typename VT>
static void afn_precond_apply(afn_precond_p ap, const int nvec, const VT *B, const int ldB, VT *C, const int ldC)
{
    if (ap == NULL) return;
    int n     = ap->n;
    int n1    = ap->n1;
    int n2    = ap->n2;
    int *perm = ap->perm;
    VT  *L11  = (VT *) ap->L11;
    VT  *K12  = (VT *) ap->K12;

    int ldB_ = n, ldC_ = n, ldT1 = n, ldT2 = n;
    VT *pB = (VT *) malloc(sizeof(VT) * n * nvec);
    VT *pC = (VT *) malloc(sizeof(VT) * n * nvec);
    VT *T1 = (VT *) malloc(sizeof(VT) * n * nvec);
    VT *T2 = (VT *) malloc(sizeof(VT) * n * nvec);
    ASSERT_PRINTF(
        pB != NULL && pC != NULL && T1 != NULL && T2 != NULL,
        "Failed to allocate work buffers for %s\n", __FUNCTION__
    );

    // 1. Permute the rows of B
    // pB = B(perm, :);
    #pragma omp parallel for schedule(static)
    for (int j = 0; j < nvec; j++)
    {
        const VT *B_j = B + j * ldB;
        VT *pB_j = pB + j * ldB_;
        #pragma omp simd
        for (int i = 0; i < n; i++) pB_j[i] = B_j[perm[i]];
    }

    // 2. Apply the preconditioner
    // B1, C1, T11, T21 are the first n1 rows of pB, pC, T1, T2
    // B2, C2, T12, T22 are the last  n2 rows of pB, pC, T1, T2
    VT *B1  = pB, *B2  = pB + n1;
    VT *C1  = pC, *C2  = pC + n1;
    VT *T11 = T1, *T12 = T1 + n1;
    VT *T21 = T2;
    VT v_neg_one = -1.0, v_one = 1.0;
    int use_omp = 1;

    // T11 = L11' \ (L11 \ B1);    % Size n1 * nvec
    // copy_matrix works on row-major matrices
    copy_matrix(sizeof(VT), nvec, n1, B1, ldB_, T11, ldT1, use_omp);
    xtrsm_(left, lower, notrans, nonunit, &n1, &nvec, &v_one, L11, &n1, T11, &ldT1);
    xtrsm_(left, lower, trans,   nonunit, &n1, &nvec, &v_one, L11, &n1, T11, &ldT1);

    // T12 = B2 - K12' * T11;      % Size n2 * nvec
    copy_matrix(sizeof(VT), nvec, n2, B2, ldB_, T12, ldT1, use_omp);
    xgemm_(
        trans, notrans, &n2, &nvec, &n1,
        &v_neg_one, K12, &n1, T11, &ldT1,
        &v_one, T12, &ldT1
    );
    
    // C2 = G * G' * T12;
    fsai_precond_apply(ap->fp, nvec, T12, ldT1, C2, ldC_);
    
    // T21 = B1 - K12 * C2;        % Size n1 * nvec
    copy_matrix(sizeof(VT), nvec, n1, B1, ldB_, T21, ldT2, use_omp);
    xgemm_(
        notrans, notrans, &n1, &nvec, &n2,
        &v_neg_one, K12, &n1, C2, &ldC_,
        &v_one, T21, &ldT2
    );

    // C1  = L11' \ (L11 \ T21);   % Size n1 * nvec
    copy_matrix(sizeof(VT), nvec, n1, T21, ldT2, C1, ldC_, use_omp);
    xtrsm_(left, lower, notrans, nonunit, &n1, &nvec, &v_one, L11, &n1, C1, &ldC_);
    xtrsm_(left, lower, trans,   nonunit, &n1, &nvec, &v_one, L11, &n1, C1, &ldC_);

    // 3. Permute back the rows of C
    // C(perm, :) = pC;
    #pragma omp parallel for schedule(static)
    for (int j = 0; j < nvec; j++)
    {
        VT *pC_j = pC + j * ldC_;
        VT *C_j  = C  + j * ldC;
        for (int i = 0; i < n; i++) C_j[perm[i]] = pC_j[i];
    }

    free(pB);
    free(pC);
    free(T1);
    free(T2);
}

void afn_precond_apply(const void *ap, const int n, const void *B, const int ldB, void *C, const int ldC)
{
    const afn_precond_p ap_ = (const afn_precond_p) ap;
    if (ap_->is_nys == 1)
    {
        nys_precond_apply(ap_->np, n, B, ldB, C, ldC);
        return;
    }
    if (ap_->val_type == VAL_TYPE_DOUBLE)
        afn_precond_apply<double>(ap_, n, (const double *) B, ldB, (double *) C, ldC);
    if (ap_->val_type == VAL_TYPE_FLOAT)
        afn_precond_apply<float> (ap_, n, (const float *)  B, ldB, (float *)  C, ldC);
}

template<typename VT>
void afn_precond_dapply(afn_precond_p ap, const int nvec, const VT *X, const int ldX, VT *Y, const int ldY)
{
    int n_grad = 3;
    int n      = ap->n;
    int n1     = ap->n1;
    int n2     = ap->n2;
    int *perm  = ap->perm;
    VT  *L11   = (VT *) ap->L11;
    VT  *K12   = (VT *) ap->K12;
    VT  *dL11  = (VT *) ap->dL11;
    VT  *dK12  = (VT *) ap->dK12;
    csr_mat_p G    = ap->fp->G;
    csr_mat_p GT   = ap->fp->GT;
    csr_mat_p *dG  = ap->fp->dG;
    csr_mat_p *dGT = ap->fp->dGT;
    VT v_zero = 0.0, v_one = 1.0;

    // 1. Permute the rows of X
    // pX = X(perm, :);
    VT *pX = (VT *) malloc(sizeof(VT) * n * nvec);
    const int ldpX = n;
    ASSERT_PRINTF(pX != NULL, "Failed to allocate work buffer for %s\n", __FUNCTION__);
    #pragma omp parallel for schedule(static)
    for (int j = 0; j < nvec; j++)
    {
        const VT *X_j = X + j * ldX;
        VT *pX_j = pX + j * ldpX;
        #pragma omp simd
        for (int i = 0; i < n; i++) pX_j[i] = X_j[perm[i]];
    }
    // Xu = pX(1 : n1, :);
    // Xl = pX(n1+ 1 : n, :);
    VT *Xu = pX, *Xl = pX + n1;
    const int ldXu = n, ldXl = n;

    // 2. GEMMs and TRSMs
    VT *iL11_K12 = (VT *) malloc(sizeof(VT) * n1 * n2);
    VT *LK_Xl    = (VT *) malloc(sizeof(VT) * n1 * nvec);
    VT *Z1u      = (VT *) malloc(sizeof(VT) * n1 * nvec);
    VT *Y1l_t0   = (VT *) malloc(sizeof(VT) * n1 * nvec);
    ASSERT_PRINTF(
        iL11_K12 != NULL && LK_Xl != NULL && Z1u != NULL && Y1l_t0 != NULL,
        "Failed to allocate work buffers for %s\n", __FUNCTION__
    );
    // iL11_K12 = L11 \ K12;
    memcpy(iL11_K12, K12, sizeof(VT) * n1 * n2);
    xtrsm_(left, lower, notrans, nonunit, &n1, &n2, &v_one, L11, &n1, iL11_K12, &n1);
    // LK_Xl    = iL11_K12 * Xl;
    xgemm_(
        notrans, notrans, &n1, &nvec, &n2,
        &v_one, iL11_K12, &n1, Xl, &ldXl,
        &v_zero, LK_Xl, &n1
    );
    // Z1u      = L11' * Xu + LK_Xl;
    memcpy(Z1u, LK_Xl, sizeof(VT) * n1 * nvec);
    xgemm_(
        trans, notrans, &n1, &nvec, &n1,
        &v_one, L11, &n1, Xu, &ldXu,
        &v_one, Z1u, &n1
    );
    // Y1l_t0   = L11' \ Z1u;
    memcpy(Y1l_t0, Z1u, sizeof(VT) * n1 * nvec);
    xtrsm_(left, lower, trans, nonunit, &n1, &nvec, &v_one, L11, &n1, Y1l_t0, &n1);

    // 3. Batched SpTRSM for Y1 and Y2
    // Z1l = G' \ Xl;
    // iG_Z1l = G \ Z1l;
    VT *Z1l      = (VT *) malloc(sizeof(VT) * n2 * nvec);
    VT *iG_Z1l   = (VT *) malloc(sizeof(VT) * n2 * nvec);
    ASSERT_PRINTF(Z1l != NULL && iG_Z1l != NULL, "Failed to allocate work buffers for %s\n", __FUNCTION__);
    csr_trsm(upper, GT, nvec, Xl,  ldXl, Z1l,    n2);
    csr_trsm(lower, G,  nvec, Z1l, n2,   iG_Z1l, n2);
    VT *tmpM132 = (VT *) malloc(sizeof(VT) * n2 * nvec * n_grad * 3);
    VT *tmpM    = (VT *) malloc(sizeof(VT) * n2 * nvec * n_grad * 2);
    ASSERT_PRINTF(
        tmpM132 != NULL && tmpM != NULL, 
        "Failed to allocate work buffers for %s\n", __FUNCTION__
    );
    VT *tmpM1 = tmpM132;
    VT *tmpM3 = tmpM1 + n2 * nvec * n_grad;
    VT *tmpM2 = tmpM3 + n2 * nvec * n_grad;
    for (int i_grad = 0; i_grad < n_grad; i_grad++)
    {
        // idx = (i - 1) * nvec + 1 : i * nvec;
        VT *tmpM1_i = tmpM1 + i_grad * n2 * nvec;
        VT *tmpM2_i = tmpM2 + i_grad * n2 * nvec;
        // tmpM1(:, idx) = dG{i} * iG_Z1l;
        // tmpM2(:, idx) = dG{i}' * Z1l;
        csr_spmm(dG[i_grad],  nvec, iG_Z1l, n2, tmpM1_i, n2);
        csr_spmm(dGT[i_grad], nvec, Z1l,    n2, tmpM2_i, n2);
    }
    // tmpM3 = G' \ tmpM2;
    // tmpM4 = [tmpM1, tmpM3];
    // tmpM  = G \ tmpM4;
    VT *tmpM4 = tmpM132;
    csr_trsm(upper, GT, n_grad * nvec,     tmpM2, n2, tmpM3, n2);
    csr_trsm(lower, G,  n_grad * nvec * 2, tmpM4, n2, tmpM,  n2);

    // 4. Compute Y1
    VT *Y1 = (VT *) malloc(sizeof(VT) * n * nvec * n_grad);
    VT *Y1l_tmp = (VT *) malloc(sizeof(VT) * (n2 * nvec * 2 + n1 * nvec));
    ASSERT_PRINTF(Y1 != NULL && Y1l_tmp != NULL, "Failed to allocate work buffers for %s\n", __FUNCTION__);
    VT *Y1l_t1  = Y1l_tmp;
    VT *Y1l_t2  = Y1l_t1 + n2 * nvec;
    VT *Y1l_t20 = Y1l_t2 + n2 * nvec;
    for (int i_grad = 0; i_grad < n_grad; i_grad++)
    {
        VT *dL11_i = dL11  + i_grad * n1 * n1;
        VT *dK12_i = dK12  + i_grad * n1 * n2;
        VT *Y1u_i  = Y1    + i_grad * n  * nvec;
        VT *Y1l_i  = Y1u_i + n1;
        const int ld_Y1u = n, ld_Y1l = n;
        // Y1u{i} = dL11{i} * Z1u;
        xgemm_(
            notrans, notrans, &n1, &nvec, &n1,
            &v_one, dL11_i, &n1, Z1u, &n1,
            &v_zero, Y1u_i, &ld_Y1u
        );
        // Y1l_t1 = dK12{i}' * Y1l_t0;
        xgemm_(
            trans, notrans, &n2, &nvec, &n1,
            &v_one, dK12_i, &n1, Y1l_t0, &n1,
            &v_zero, Y1l_t1, &n2
        );
        // Y1l_t2 = iL11_K12' * (dL11{i}' * Y1l_t0);
        xgemm_(
            trans, notrans, &n1, &nvec, &n1,
            &v_one, dL11_i, &n1, Y1l_t0, &n1,
            &v_zero, Y1l_t20, &n1
        );
        xgemm_(
            trans, notrans, &n2, &nvec, &n1,
            &v_one, iL11_K12, &n1, Y1l_t20, &n1,
            &v_zero, Y1l_t2, &n2
        );
        // idx = (i - 1) * nvec + 1 : i * nvec;
        // Y1l_t3 = tmpM(:, idx);
        VT *Y1l_t3 = tmpM + i_grad * n2 * nvec;
        // Y1l{i} = Y1l_t1 - Y1l_t2 - Y1l_t3;
        #pragma omp parallel for schedule(static)
        for (int j = 0; j < nvec; j++)
        {
            VT *Y1l_i_j  = Y1l_i  + j * ld_Y1l;
            VT *Y1l_t1_j = Y1l_t1 + j * n2;
            VT *Y1l_t2_j = Y1l_t2 + j * n2;
            VT *Y1l_t3_j = Y1l_t3 + j * n2;
            #pragma omp simd
            for (int k = 0; k < n2; k++)
                Y1l_i_j[k] = Y1l_t1_j[k] - Y1l_t2_j[k] - Y1l_t3_j[k];
        }  // End of j loop
    }  // End of i_grad loop
    
    // 5. Compute Y2
    VT *Y2 = (VT *) malloc(sizeof(VT) * n * nvec * n_grad);
    VT *Z2u_tmp = (VT *) malloc(sizeof(VT) * n1 * nvec * 4);
    ASSERT_PRINTF(Y2 != NULL && Z2u_tmp != NULL, "Failed to allocate work buffers for %s\n", __FUNCTION__);
    VT *Z2u_t0 = Z2u_tmp;
    VT *Z2u_t1 = Z2u_t0 + n1 * nvec;
    VT *Z2u_t2 = Z2u_t1 + n1 * nvec;
    VT *Z2u    = Z2u_t2 + n1 * nvec;
    for (int i_grad = 0; i_grad < n_grad; i_grad++)
    {
        // idx = (i - 1) * nvec + 1 : i * nvec;
        // idx = idx + n_grad * nvec;
        // iG_Z2l = tmpM(:, idx);
        // Y2l{i} = -iG_Z2l;
        VT *dL11_i = dL11 + i_grad * n1 * n1;
        VT *dK12_i = dK12 + i_grad * n1 * n2;
        VT *iG_Z2l = tmpM + (n_grad + i_grad) * n2 * nvec;
        VT *Y2u_i  = Y2 + i_grad * n * nvec;
        VT *Y2l_i  = Y2u_i + n1;
        const int ld_Y2u = n, ld_Y2l = n;
        #pragma omp parallel for schedule(static)
        for (int j = 0; j < nvec; j++)
        {
            VT *Y2l_i_j  = Y2l_i  + j * ld_Y2l;
            VT *iG_Z2l_j = iG_Z2l + j * n2;
            #pragma omp simd
            for (int k = 0; k < n2; k++) Y2l_i_j[k] = -iG_Z2l_j[k];
        }
        // Z2u_t0 = dK12{i} * Xl;
        xgemm_(
            notrans, notrans, &n1, &nvec, &n2,
            &v_one, dK12_i, &n1, Xl, &ldXl,
            &v_zero, Z2u_t0, &n1
        ); 
        // Z2u_t1 = dL11{i} * LK_Xl;
        xgemm_(
            notrans, notrans, &n1, &nvec, &n1,
            &v_one, dL11_i, &n1, LK_Xl, &n1,
            &v_zero, Z2u_t1, &n1
        );
        // Z2u_t2 = Z2u_t0 - Z2u_t1;
        for (int i = 0; i < n1 * nvec; i++) Z2u_t2[i] = Z2u_t0[i] - Z2u_t1[i];
        // Z2u    = (dL11{i}' * Xu) + L11 \ Z2u_t2;
        xtrsm_(left, lower, notrans, nonunit, &n1, &nvec, &v_one, L11, &n1, Z2u_t2, &n1);
        memcpy(Z2u, Z2u_t2, sizeof(VT) * n1 * nvec);
        xgemm_(
            trans, notrans, &n1, &nvec, &n1,
            &v_one, dL11_i, &n1, Xu, &ldXu,
            &v_one, Z2u, &n1
        );
        // Y2u{i} = L11 * Z2u;
        xgemm_(
            notrans, notrans, &n1, &nvec, &n1,
            &v_one, L11, &n1, Z2u, &n1,
            &v_zero, Y2u_i, &ld_Y2u
        );
        // Y2l{i} = Y2l{i} + iL11_K12' * Z2u;
        xgemm_(
            trans, notrans, &n2, &nvec, &n1,
            &v_one, iL11_K12, &n1, Z2u, &n1,
            &v_one, Y2l_i, &ld_Y2l
        );
    }  // End of i_grad loop

    // 6. Get the final Y
    // Y1 = Y1 + Y2
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n * nvec * n_grad; i++) Y1[i] += Y2[i];
    // Y{i}(perm, :) = [Y1u{i}; Y1l{i}];
    for (int i_grad = 0; i_grad < n_grad; i_grad++)
    {
        VT *Y_i  = Y  + ldY * nvec * i_grad;
        VT *Y1_i = Y1 + n   * nvec * i_grad;
        #pragma omp parallel for schedule(static)
        for (int j = 0; j < nvec; j++)
        {
            VT *Y_i_j  = Y_i  + j * ldY;
            VT *Y1_i_j = Y1_i + j * n;
            #pragma omp simd
            for (int k = 0; k < n; k++) Y_i_j[perm[k]] = Y1_i_j[k];
        }
    }

    // Free work arrays
    free(pX);
    free(iL11_K12);
    free(LK_Xl);
    free(Z1u);
    free(Y1l_t0);
    free(Z1l);
    free(iG_Z1l);
    free(tmpM132);
    free(tmpM);
    free(Y1);
    free(Y1l_tmp);
    free(Y2);
    free(Z2u_tmp);
}

// Compute Y := d M^{-1} / d {l, f, s} * X, M is the AFN precond matrix
void afn_precond_dapply(afn_precond_p ap, const int n, const void *X, const int ldX, void *Y, const int ldY)
{
    if (ap->is_nys == 1)
    {
        nys_precond_dapply(ap->np, n, X, ldX, Y, ldY, 0);
        return;
    }
    if (ap->val_type == VAL_TYPE_DOUBLE)
        afn_precond_dapply<double>(ap, n, (const double *) X, ldX, (double *) Y, ldY);
    if (ap->val_type == VAL_TYPE_FLOAT)
        afn_precond_dapply<float> (ap, n, (const float *)  X, ldX, (float *)  Y, ldY);
}

template<typename VT>
static void afn_precond_grad_trace(afn_precond_p ap, VT *gt)
{
    int n_grad = 3;
    int n = ap->n, n1 = ap->n1, n2 = ap->n2;
    VT *diagM     = (VT *) malloc(sizeof(VT) * n);
    VT *inv_diagG = (VT *) malloc(sizeof(VT) * n2);
    VT *diag_dMi  = (VT *) malloc(sizeof(VT) * n);
    ASSERT_PRINTF(
        diagM != NULL && inv_diagG != NULL && diag_dMi != NULL,
        "Failed to allocate work arrays for %s\n", __FUNCTION__
    );
    // diagM(1 : n1) = diag(ap.L11);
    VT *L11 = (VT *) ap->L11;
    for (int i = 0; i < n1; i++) diagM[i] = L11[i * n1 + i];
    // inv_diagG = 1 ./ diag(ap.G);
    // diagM(n1+1 : n) = inv_diagG;
    int *G_rowptr = ap->fp->G->rowptr;
    VT *G_val = (VT *) ap->fp->G->val;
    for (int i = 0; i < n2; i++)
    {
        int G_ii_idx = G_rowptr[i + 1] - 1;
        inv_diagG[i] = 1.0 / G_val[G_ii_idx];
        diagM[n1 + i] = inv_diagG[i];
    }
    for (int i = 0; i < n_grad; i++)
    {
        VT *dL11_i = ((VT *) ap->dL11) + i * n1 * n1;
        int *dGi_rowptr = ap->fp->dG[i]->rowptr;
        VT *dGi_val = (VT *) ap->fp->dG[i]->val;
        // diag_dMi(1 : n1) = diag(ap.dL11{i});
        for (int j = 0; j < n1; j++) diag_dMi[j] = dL11_i[j * n1 + j];
        // diag_dMi(n1+1 : n) = -inv_diagG.^2 .* diag(ap.dG{i});
        for (int j = 0; j < n2; j++)
        {
            int dGi_jj_idx = dGi_rowptr[j + 1] - 1;
            diag_dMi[n1 + j] = -inv_diagG[j] * inv_diagG[j] * dGi_val[dGi_jj_idx];
        }
        // gt(i) = 2 * sum(diag_dMi ./ diagM);
        gt[i] = 0.0;
        for (int j = 0; j < n; j++) gt[i] += diag_dMi[j] / diagM[j];
        gt[i] *= 2.0;
    }
    free(diagM);
    free(inv_diagG);
    free(diag_dMi);
}

// Compute the trace of the AFN gradient matrices dM^{-1}/dx
void afn_precond_grad_trace(afn_precond_p ap)
{
    if (ap == NULL) return;
    if (ap->gt != NULL) return;
    size_t VT_bytes = (ap->val_type == VAL_TYPE_DOUBLE) ? sizeof(double) : sizeof(float);
    ap->gt = (void *) malloc(VT_bytes * 3);
    if (ap->val_type == VAL_TYPE_FLOAT)   afn_precond_grad_trace<float> (ap, (float *)  ap->gt);
    if (ap->val_type == VAL_TYPE_DOUBLE)  afn_precond_grad_trace<double>(ap, (double *) ap->gt);
}