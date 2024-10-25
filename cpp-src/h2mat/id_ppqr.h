#ifndef __ID_PPQR_H__
#define __ID_PPQR_H__

#ifdef __cplusplus
extern "C" {
#endif

// Compute an Interpolative Decomposition (ID) of a given matrix A
// A ~= A(:, col_idx) * V using partial pivoting QR factorization
// Input parameters:
//   nrow, ncol : Number of rows and columns of A
//   A, ldA     : Size ldA * ncol, col-major, matrix to be decomposed, ldA >= nrow
//   max_rank   : Maximum rank of the ID, if <= 0 means no limitation
//   *rel_tol   : Relative tolerance of the ID, if <= 0 means no limitation
//   n_thread   : Number of threads to use, if <= 0 means using all available threads
// Output parameters:
//   A          : The original matrix A is overwritten by intermediate results
//   *rank      : Rank of the ID
//   *col_idx   : Size (*rank), ID selected column indices, will be allocated in this function
//   *V         : Size (*rank) * ncol, col-major, ID basis, will be allocated in this function
//   *worki     : Size 4 * ncol, integer work buffer, if == NULL will be allocated internally
//   *workv     : Size nrow * ncol, float/double work buffer, if == NULL will be allocated internally
// Note: max_rank and rel_tol cannot be both <= 0
void id_ppqr(
    const int nrow, const int ncol, const int val_type, void *A, const int ldA, 
    const int max_rank, const void *rel_tol, const int n_thread, 
    int *rank, int **col_idx, void **V, int *worki, void *workv
);

#ifdef __cplusplus
}
#endif

#endif
