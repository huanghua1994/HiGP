#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <type_traits>
#include <omp.h>

#include "id_ppqr.h"
#include "../common.h"
#include "../cpu_linalg.hpp"

// Partial pivoting QR factorization with simplified output
// The partial pivoting QR decomposition is of form:
//     A * P = Q * [R11, R12; 0, R22]
// where R11 is an upper-triangular matrix, R12 and R22 are dense matrices,
// P is a permutation matrix. 
// Input parameters:
//   nrow, ncol : Number of rows and columns of A
//   A, ldA     : Size ldA * ncol, col-major, matrix to be decomposed, ldA >= nrow
//   max_rank   : Maximum rank of the ID, must > 0
//   rel_tol    : Relative tolerance of the ID, should be 0 < rel_tol < 1
//   n_thread   : Number of threads to use, if <= 0 means using all available threads
//   col_norm   : Size ncol, work buffer, for storing column norms of A
// Output parameters:
//   A          : The original matrix A is overwritten by R: [R11, R12; 0, R22]
//   rank       : Rank of the ID, == size of R11
//   p          : Size ncol, A column permutation array, A(:, p) = A * P
template<typename VT>
static void ppqr(
    const int nrow, const int ncol, VT *A, const int ldA,
    const int max_rank, const VT rel_tol, int *rank, int *p, 
    const int n_thread, VT *col_norm
)
{
    bool VT_is_float  = std::is_same<VT, float>::value;
    bool VT_is_double = std::is_same<VT, double>::value;
    int i_one = 1;

    VT *R = A;
    int ldR = ldA, max_iter = std::min(nrow, ncol);
    VT VT_eps, fast_norm_threshold;
    if (VT_is_float)
    {
        VT_eps = 1e-6;
        fast_norm_threshold = 1e-5;
    }
    if (VT_is_double)
    {
        VT_eps = 1e-15;
        fast_norm_threshold = 1e-12;
    }

    // Find a column with largest 2-norm
    #pragma omp parallel if(n_thread > 1) num_threads(n_thread)
    #pragma omp for schedule(static)
    for (int j = 0; j < ncol; j++)
    {
        p[j] = j;
        VT *R_j = R + j * ldR;
        col_norm[j] = xnrm2_(&nrow, R_j, &i_one);
    }
    int pivot = std::max_element(col_norm, col_norm + ncol) - col_norm;
    VT norm_p = col_norm[pivot];

    // Scale the stopping norm
    int stop_rank = std::min(max_iter, max_rank);
    VT  stop_norm = std::max(VT_eps, norm_p * rel_tol) * 0.5;

    int rank_ = -1;
    // Main iteration of Household QR
    for (int i = 0; i < max_iter; i++)
    {   
        // 1. Check the stop criteria
        if ((norm_p < stop_norm) || (i >= stop_rank))
        {
            rank_ = i;
            break;
        }
        
        // 2. Swap the column
        if (i != pivot)
        {
            std::swap(p[i], p[pivot]);
            std::swap(col_norm[i], col_norm[pivot]);
            for (int j = 0; j < nrow; j++) std::swap(R[i * ldR + j], R[pivot * ldR + j]);
        }
        
        // 3. Calculate Householder vector
        int h_len    = nrow - i;
        int h_len_m1 = h_len - 1;
        VT *h_vec = R + i * ldR + i;
        VT h_norm, inv_h_norm, sign;
        sign = (h_vec[0] > 0.0) ? 1.0 : -1.0;
        h_norm = xnrm2_(&h_len, h_vec, &i_one);
        h_vec[0] = h_vec[0] + sign * h_norm;
        inv_h_norm = 1.0 / xnrm2_(&h_len, h_vec, &i_one);
        #pragma omp simd
        for (int j = 0; j < h_len; j++) h_vec[j] *= inv_h_norm;
        
        // 4. & 5. Householder update & column norm update
        VT *R_block = R + (i + 1) * ldR + i;
        int R_block_nrow = h_len;
        int R_block_ncol = ncol - i - 1;
        #pragma omp parallel if(n_thread > 1) num_threads(n_thread)
        #pragma omp for schedule(guided)
        for (int j = 0; j < R_block_ncol; j++)
        {
            int ji1 = j + i + 1;
            
            VT *R_block_j = R_block + j * ldR;
            VT h_Rj = 2.0 * xdot_(&R_block_nrow, h_vec, &i_one, R_block_j, &i_one);
            
            // 4. Orthogonalize columns right to the i-th column
            #pragma omp simd
            for (int k = 0; k < R_block_nrow; k++) R_block_j[k] -= h_Rj * h_vec[k];
            
            // 5. Update i-th column's 2-norm
            if (col_norm[ji1] < stop_norm)
            {
                col_norm[ji1] = 0.0;
                continue;
            }
            VT tmp = R_block_j[0] * R_block_j[0];
            tmp = col_norm[ji1] * col_norm[ji1] - tmp;
            if (tmp <= fast_norm_threshold)
            {
                VT *R_block_j1 = R_block_j + 1;
                col_norm[ji1] = xnrm2_(&h_len_m1, R_block_j1, &i_one);
            } else {
                // Fast update 2-norm when the new column norm is not so small
                col_norm[ji1] = std::sqrt(tmp);
            }
        }  // End of j loop
        
        // We don't need h_vec anymore, can overwrite the i-th column of R
        h_vec[0] = -sign * h_norm;
        memset(h_vec + 1, 0, sizeof(VT) * (h_len - 1));
        // Find next pivot
        pivot = std::max_element(col_norm + i + 1, col_norm + ncol) - col_norm;
        norm_p = col_norm[pivot];
    }  // End of i loop
    if (rank_ == -1) rank_ = max_iter;
    *rank = rank_;
}

template<typename VT>
static void id_ppqr(
    const int nrow, const int ncol, const int val_type, VT *A, const int ldA, 
    const int max_rank, const VT *rel_tol_, const int n_thread, 
    int *rank, int **col_idx, VT **V, int *worki, VT *workv
)
{
    const VT rel_tol = *rel_tol_;
    int *worki_ = worki;
    VT  *workv_ = workv;
    if (worki_ == NULL) worki_ = (int *) malloc(sizeof(int) * ncol * 4);
    if (workv_ == NULL) workv_ = (VT  *) malloc(sizeof(VT) * nrow * ncol);

    // 1. Partial pivoting QR of A
    int r = 0, *p = (int *) malloc(sizeof(int) * ncol);
    int n_thread_ = (n_thread < 1) ? omp_get_max_threads() : n_thread;
    int max_rank_ = (max_rank < 1) ? std::min(nrow, ncol) : max_rank;
    ppqr<VT>(
        nrow, ncol, A, ldA, max_rank_, rel_tol,
        &r, p, n_thread_, workv_
    );

    // 2. Set permutation indices p0, sort the index subset p[0 : r-1]
    int *p0 = worki_;
    int *p1 = worki_ + ncol;
    int *i0 = worki_ + ncol * 2;
    int *i1 = worki_ + ncol * 3;
    for (int i = 0; i < ncol; i++) 
    {
        p0[p[i]] = i;
        i0[i] = i;
    }
    qsort_key_val<int, int>(p, i0, 0, r - 1);
    for (int i = 0; i < ncol; i++) i1[i0[i]] = i;
    for (int i = 0; i < ncol; i++) p1[i] = i1[p0[i]];

    // 3. Construct the V matrix
    // (1) Solve E = inv(R11) * R12, E is stored in R12 in column major style
    VT *R11 = A, *R12 = A + r * ldA;
    int ldR = ldA, nrow_R12 = r, ncol_R12 = ncol - r;
    VT v_one = 1.0;
    xtrsm_(left, upper, notrans, nonunit, &nrow_R12, &ncol_R12, &v_one, R11, &ldR, R12, &ldR); 
    // (2) E1 = E(i0, :);  V0 = [eye(size(E, 1)), E1];
    VT *V0_ = workv_;
    VT *E1  = V0_ + r * r;
    for (int i = 0; i < r; i++)
    {
        memset(V0_ + i * r, 0, sizeof(VT) * r);
        V0_[i * r + i] = 1.0;
    }
    for (int irow = 0; irow < r; irow++)
    {
        VT *E_irow  = R12 + i0[irow];
        VT *E1_irow = E1  + irow;
        for (int icol = 0; icol < ncol_R12; icol++)
            E1_irow[icol * r] = E_irow[icol * ldR];
    }
    // (3) V = V0(:, p1);
    VT *V_ = (VT *) malloc(sizeof(VT) * r * ncol);
    for (int i = 0; i < ncol; i++)
        memcpy(V_ + i * r, V0_ + p1[i] * r, sizeof(VT) * r);

    // 4. Set output parameters and free work arrays
    *rank = r;
    *col_idx = p;
    *V = V_;
    if (worki == NULL) free(worki_);
    if (workv == NULL) free(workv_);
}

void id_ppqr(
    const int nrow, const int ncol, const int val_type, void *A, const int ldA, 
    const int max_rank, const void *rel_tol, const int n_thread, 
    int *rank, int **col_idx, void **V, int *worki, void *workv
)
{
    if (val_type == VAL_TYPE_DOUBLE)
    {
        id_ppqr<double>(
            nrow, ncol, val_type, (double *) A, ldA, 
            max_rank, (const double *) rel_tol, n_thread, 
            rank, col_idx, (double **) V, worki, (double *) workv
        );
    }
    if (val_type == VAL_TYPE_FLOAT)
    {
        id_ppqr<float>(
            nrow, ncol, val_type, (float *)  A, ldA, 
            max_rank, (const float *)  rel_tol, n_thread, 
            rank, col_idx, (float **)  V, worki, (float *)  workv
        );
    }
}
