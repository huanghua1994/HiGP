#ifndef __CSR_MAT_H__
#define __CSR_MAT_H__


#include "../common.h"

struct csr_mat {
    int  val_type;
    int  nrow, ncol, nnz;
    int  *rowptr;
    int  *colidx;
    void *val;

    // For parallel TRSM only
    int trsm_nlevel;    // Number of levels in the TRSM tree
    int *trsm_lvlptr;   // Size trsm_nlevel + 1, index of the first row in each level
    int *trsm_rowidx;   // Size nrow, row indices of the TRSM tree
};
typedef struct csr_mat  csr_mat_s;
typedef struct csr_mat *csr_mat_p;

#ifdef __cplusplus
extern "C" {
#endif

// Convert a COO matrix to a sorted CSR matrix
// Input parameters:
//   val_type        : 0 for double, 1 for float, 2 for int
//   nrow, ncol, nnz : Number of rows, columns, non-zeros
//   row, col, val   : Size nnz, COO matrix
// Output parameter:
//   csr_mat : CSR matrix
void coo_to_csr(
    const int val_type, const int nrow, const int ncol, const int nnz,
    const int *row, const int *col, const void *val, csr_mat_p *csr_mat
);

// build a CSR matrix from a precomputed sparsity pattern
// Input parameters:
//   val_type : 0 for double, 1 for float
//   M_pat    : CSR matrix sparsity pattern matrix, with val array being a permutation
//   val      : Size M_pat->nnz, values of the CSR matrix
// Output parameter:
//   M : CSR matrix
void csr_build_from_pattern(const int val_type, const csr_mat_p M_pat, const void *val, csr_mat_p *M);

// CSR SpMM Y := A * X
// Input parameters:
//   cst_mat : CSR matrix
//   n       : Number of columns of X and Y
//   X       : Input dense matrix, size ldX * n, col-major
//   ldX     : Leading dimension of X, >= k
//   ldY     : Leading dimension of Y, >= m
// Output parameters:
//   Y : Output dense matrix, size ldY * n, col-major
void csr_spmm(const csr_mat_p csr_mat, const int n, const void *X, const int ldX, void *Y, const int ldY);

// Build a TRSM dependency tree for a triangular matrix
// Input parameters:
//   *uplo : "L" for lower triangular, "U" for upper triangular
//   M     : A lower or upper triangular matrix in CSR format
// Output parameter:
//   M : Updated with TRSM tree information
void csr_trsm_build_tree(const char *uplo, csr_mat_p M);

// Solve a triangular system L * X = B or U * X = B
// Input parameters:
//   *uplo : "L" for lower triangular, "U" for upper triangular
//   M     : A lower or upper triangular matrix in CSR format
//   n     : Number of columns of X and B
//   B     : Input dense matrix, size ldB * n, col-major
//   ldB   : Leading dimension of B, >= {L, U}->nrow
//   ldX   : Leading dimension of X, >= {L, U}->nrow
// Output parameter:
//   X : Output dense matrix, size ldX * n, col-major
// Note: L / U must have diagonal elements one the last / first nnz positions of each row
void csr_trsm(
    const char *uplo, const csr_mat_p M, const int n, 
    const void *B, const int ldB, void *X, const int ldX
);

// Free a csr_mat structure, set the pointer to NULL
void csr_mat_free(csr_mat_p *csr_mat);

#ifdef __cplusplus
}
#endif

#endif
