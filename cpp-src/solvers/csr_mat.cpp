#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <type_traits>
#include <limits>
#include <omp.h>

#include "csr_mat.h"
#include "../utils.h"

template<typename VT>
static void coo_to_csr(
    const int val_type, const int nrow, const int ncol, const int nnz,
    const int *row, const int *col, const VT *val, csr_mat_p *csr_mat
)
{
    int *rowptr = (int *) malloc(sizeof(int) * (nrow + 1));
    int *colidx = (int *) malloc(sizeof(int) * nnz);
    VT  *val2   = (VT *)  malloc(sizeof(VT)  * nnz);
    ASSERT_PRINTF(
        rowptr != NULL && colidx != NULL && val2 != NULL, 
        "Failed to allocate CSR arrays for matrix of size %d * %d, %d nnz\n",
        nrow, ncol, nnz
    );

    // Get the number of non-zeros in each row
    memset(rowptr, 0, sizeof(int) * (nrow + 1));
    for (int i = 0; i < nnz; i++) rowptr[row[i] + 1]++;

    // Calculate the displacement of 1st non-zero in each row
    for (int i = 2; i <= nrow; i++) rowptr[i] += rowptr[i - 1];

    // Use rowptr to bucket sort col[] and val[]
    for (int i = 0; i < nnz; i++)
    {
        int idx = rowptr[row[i]];
        colidx[idx] = col[i];
        val2[idx] = val[i];
        rowptr[row[i]]++;
    }

    // Reset rowptr
    for (int i = nrow; i >= 1; i--) rowptr[i] = rowptr[i - 1];
    rowptr[0] = 0;

    // Sort the non-zeros in each row according to column indices
    #pragma omp parallel for
    for (int i = 0; i < nrow; i++)
        qsort_key_val<int, VT>(colidx, val2, rowptr[i], rowptr[i + 1] - 1);

    // Wrap the CSR matrix
    csr_mat_p csr_mat_ = (csr_mat_p) malloc(sizeof(csr_mat_s));
    memset(csr_mat_, 0, sizeof(csr_mat_s));
    csr_mat_->val_type = val_type;
    csr_mat_->nrow     = nrow;
    csr_mat_->ncol     = ncol;
    csr_mat_->nnz      = nnz;
    csr_mat_->rowptr   = rowptr;
    csr_mat_->colidx   = colidx;
    csr_mat_->val      = val2;
    *csr_mat = csr_mat_;
}

// Convert a COO matrix to a sorted CSR matrix
void coo_to_csr(
    const int val_type, const int nrow, const int ncol, const int nnz,
    const int *row, const int *col, const void *val, csr_mat_p *csr_mat
)
{
    if (val_type == VAL_TYPE_DOUBLE)
        coo_to_csr<double>(val_type, nrow, ncol, nnz, row, col, (double *) val, csr_mat);
    if (val_type == VAL_TYPE_FLOAT)
        coo_to_csr<float> (val_type, nrow, ncol, nnz, row, col, (float *)  val, csr_mat);
    if (val_type == VAL_TYPE_INT)
        coo_to_csr<int>   (val_type, nrow, ncol, nnz, row, col, (int *)    val, csr_mat);
}

template<typename VT>
static void csr_build_from_pattern(const int val_type, const csr_mat_p M_pat, const VT *val, csr_mat_p *M)
{
    csr_mat_p M_ = (csr_mat_p) malloc(sizeof(csr_mat_s));
    memset(M_, 0, sizeof(csr_mat_s));
    M_->val_type    = val_type;
    M_->nrow        = M_pat->nrow;
    M_->ncol        = M_pat->ncol;
    M_->nnz         = M_pat->nnz;
    M_->trsm_nlevel = M_pat->trsm_nlevel;
    M_->rowptr      = (int *) malloc(sizeof(int) * (M_->nrow + 1));
    M_->colidx      = (int *) malloc(sizeof(int) * M_->nnz);
    M_->val         = (VT *)  malloc(sizeof(VT)  * M_->nnz);
    M_->trsm_lvlptr = (int *) malloc(sizeof(int) * (M_->trsm_nlevel + 1));
    M_->trsm_rowidx = (int *) malloc(sizeof(int) * M_->nrow);
    ASSERT_PRINTF(
        M_->rowptr != NULL && M_->colidx != NULL && M_->val != NULL 
        && M_->trsm_lvlptr != NULL && M_->trsm_rowidx != NULL,
        "Failed to allocate CSR arrays for matrix of size %d * %d, %d nnz\n",
        M_->nrow, M_->ncol, M_->nnz
    );
    memcpy(M_->rowptr, M_pat->rowptr, sizeof(int) * (M_->nrow + 1));
    memcpy(M_->colidx, M_pat->colidx, sizeof(int) * M_->nnz);
    memcpy(M_->trsm_lvlptr, M_pat->trsm_lvlptr, sizeof(int) * (M_->trsm_nlevel + 1));
    memcpy(M_->trsm_rowidx, M_pat->trsm_rowidx, sizeof(int) * M_->nrow);
    int *perm = (int *) M_pat->val;
    VT *M_val = (VT *) M_->val;
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < M_->nnz; i++) M_val[i] = val[perm[i]];
    *M = M_;
}

// build a CSR matrix from a precomputed sparsity pattern
void csr_build_from_pattern(const int val_type, const csr_mat_p M_pat, const void *val, csr_mat_p *M)
{
    if (val_type == VAL_TYPE_DOUBLE)
        csr_build_from_pattern<double>(val_type, M_pat, (double *) val, M);
    if (val_type == VAL_TYPE_FLOAT)
        csr_build_from_pattern<float> (val_type, M_pat, (float *)  val, M);
}

template<typename VT>
static void csr_spmm(const csr_mat_p csr_mat, const int n, const VT *X, const int ldX, VT *Y, const int ldY)
{
    int *rowptr = csr_mat->rowptr;
    int *colidx = csr_mat->colidx;
    VT  *val    = (VT *) csr_mat->val;
    // We assume the A matrix is obtained from FSAI, so it has 
    // fsai_npt nonzeros per row after the first fsai_npt rows. 
    // Using a static partitioning scheme should be good enough.
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < csr_mat->nrow; i++)
    {
        for (int j = 0; j < n; j++)
        {
            const VT *X_j = X + j * ldX;
            VT *Y_j = Y + j * ldY;
            VT sum = 0.0;
            #pragma omp simd
            for (int k = rowptr[i]; k < rowptr[i + 1]; k++)
                sum += val[k] * X_j[colidx[k]];
            Y_j[i] = sum;
        }
    }
}

// CSR SpMM Y := A * X
void csr_spmm(const csr_mat_p csr_mat, const int n, const void *X, const int ldX, void *Y, const int ldY)
{
    if (csr_mat->val_type == VAL_TYPE_DOUBLE)
        csr_spmm<double>(csr_mat, n, (const double *) X, ldX, (double *) Y, ldY);
    if (csr_mat->val_type == VAL_TYPE_FLOAT)
        csr_spmm<float> (csr_mat, n, (const float *)  X, ldX, (float *)  Y, ldY);
}

// Build a TRSM dependency tree for a triangular matrix
void csr_trsm_build_tree(const char *uplo, csr_mat_p M)
{
    int nrow = M->nrow;
    int *rowptr = M->rowptr;
    int *colidx = M->colidx;

    int nlevel = 0;
    int *lvl_cnt = (int *) malloc(sizeof(int) * nrow);
    int *row_lvl = (int *) malloc(sizeof(int) * nrow);
    memset(lvl_cnt, 0, sizeof(int) * nrow);

    if (uplo[0] == 'U')
    {
        row_lvl[nrow - 1] = 0;
        lvl_cnt[0] = 1;
        for (int i = nrow - 2; i >= 0; i--)
        {
            int my_lvl = 0;
            for (int k = rowptr[i] + 1; k < rowptr[i + 1]; k++) 
                if (row_lvl[colidx[k]] > my_lvl) my_lvl = row_lvl[colidx[k]];
            my_lvl++;
            row_lvl[i] = my_lvl;
            lvl_cnt[my_lvl]++;
            if (my_lvl > nlevel) nlevel = my_lvl;
        }  // End of i loop
    }  // End of "if (uplo[0] == 'U')"

    if (uplo[0] == 'L') 
    {
        row_lvl[0] = 0;
        lvl_cnt[0] = 1;
        for (int i = 1; i < nrow; i++)
        {
            int my_lvl = 0;
            for (int k = rowptr[i]; k < rowptr[i + 1] - 1; k++)
                if (row_lvl[colidx[k]] > my_lvl) my_lvl = row_lvl[colidx[k]];
            my_lvl++;
            row_lvl[i] = my_lvl;
            lvl_cnt[my_lvl]++;
            if (my_lvl > nlevel) nlevel = my_lvl;
        }  // End of i loop
    }  // End of "if (uplo[0] == 'L')"

    nlevel++;
    int *lvlptr = (int *) malloc(sizeof(int) * (nlevel + 1));
    int *rowidx = (int *) malloc(sizeof(int) * nrow);
    lvlptr[0] = 0;
    for (int i = 0; i < nlevel; i++) lvlptr[i + 1] = lvlptr[i] + lvl_cnt[i];
    for (int i = 0; i < nrow; i++)
    {
        int lvl_i = row_lvl[i];
        int idx = lvlptr[lvl_i];
        rowidx[idx] = i;
        lvlptr[lvl_i]++;
    }
    for (int i = nlevel; i >= 1; i--) lvlptr[i] = lvlptr[i - 1];
    lvlptr[0] = 0;
    M->trsm_nlevel = nlevel;
    M->trsm_lvlptr = lvlptr;
    M->trsm_rowidx = rowidx;

    free(lvl_cnt);
    free(row_lvl);
}

template<typename VT, bool is_upper>
static void csr_trsm(const csr_mat_p M, const int n, const VT *B, const int ldB, VT *X, const int ldX)
{
    int nlevel  = M->trsm_nlevel;
    int *lvlptr = M->trsm_lvlptr;
    int *rowidx = M->trsm_rowidx;
    int *rowptr = M->rowptr;
    int *colidx = M->colidx;
    VT  *val    = (VT *) M->val;
    copy_matrix(sizeof(VT), n, M->nrow, B, ldB, X, ldX, 1);
    for (int i_lvl = 0; i_lvl < nlevel; i_lvl++)
    {
        #pragma omp parallel for schedule(dynamic)
        for (int ii = lvlptr[i_lvl]; ii < lvlptr[i_lvl + 1]; ii++)
        {
            int i = rowidx[ii];
            int k_start, k_end;
            VT diag = 0.0;
            if (is_upper)
            {
                k_start = rowptr[i] + 1;
                k_end   = rowptr[i + 1];
                diag    = val[rowptr[i]];
            } else {
                k_start = rowptr[i];
                k_end   = rowptr[i + 1] - 1;
                diag    = val[rowptr[i + 1] - 1];
            }
            for (int j = 0; j < n; j++)
            {
                VT *X_j = X + j * ldX;
                VT sum = 0.0;
                #pragma omp simd
                for (int k = k_start; k < k_end; k++)
                    sum += val[k] * X_j[colidx[k]];
                X_j[i] = (X_j[i] - sum) / diag;
            }  // End of j loop
        }  // End of ii loop
    }  // End of i_lvl loop
}

void csr_trsm(
    const char *uplo, const csr_mat_p M, const int n, 
    const void *B, const int ldB, void *X, const int ldX
)
{
    if (M->trsm_nlevel == 0) csr_trsm_build_tree(uplo, M);
    if (uplo[0] == 'L')
    {
        if (M->val_type == VAL_TYPE_DOUBLE)
            csr_trsm<double, false>(M, n, (const double *) B, ldB, (double *) X, ldX);
        if (M->val_type == VAL_TYPE_FLOAT)
            csr_trsm<float,  false>(M, n, (const float *)  B, ldB, (float *)  X, ldX);
    }
    if (uplo[0] == 'U')
    {
        if (M->val_type == VAL_TYPE_DOUBLE)
            csr_trsm<double, true>(M, n, (const double *) B, ldB, (double *) X, ldX);
        if (M->val_type == VAL_TYPE_FLOAT)
            csr_trsm<float,  true>(M, n, (const float *)  B, ldB, (float *)  X, ldX);
    }
}

// Free a csr_mat structure, set the pointer to NULL
void csr_mat_free(csr_mat_p *csr_mat)
{
    csr_mat_p csr_mat_ = *csr_mat;
    if (csr_mat_ == NULL) return;
    free(csr_mat_->rowptr);
    free(csr_mat_->colidx);
    free(csr_mat_->val);
    free(csr_mat_->trsm_lvlptr);
    free(csr_mat_->trsm_rowidx);
    free(csr_mat_);
    *csr_mat = NULL;
}
