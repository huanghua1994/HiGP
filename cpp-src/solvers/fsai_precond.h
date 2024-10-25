#ifndef __FSAI_PRECOND_H__
#define __FSAI_PRECOND_H__

#include "csr_mat.h"
#include "../h2mat/octree.h"

struct fsai_precond
{
    int n;              // Size of the kernel matrix
    int val_type;       // Type of the kernel matrix values, 0: double, 1: float
    csr_mat_p G;        // G matrix, lower triangle sparse matrix
    csr_mat_p GT;       // G^T matrix, upper triangle sparse matrix
    csr_mat_p *dG;      // Size 3, d G / d {l, f, s} matrices
    csr_mat_p *dGT;     // Size 3, d G^T / d {l, f, s} matrices
};
typedef struct fsai_precond  fsai_precond_s;
typedef struct fsai_precond *fsai_precond_p;

#ifdef __cplusplus
extern "C" {
#endif

// Select k exact nearest neighbors for each point s.t. the indices of 
// neighbors are smaller than the index of the point
// Input parameters:
//   val_type  : Type of point coordinate, 0: double, 1: float
//   fsai_npt  : Maximum number of nonzeros in each row of the FSAI matrix (number of nearest neighbors)
//   n, pt_dim : Number of points and point dimension
//   coord     : Size pt_dim * ldc, row major, each column is a point coordinate
//   ldc       : Leading dimension of coord, >= n
// Output parameters:
//   nn_idx : Size n * fsai_npt, row major, indices of the nearest neighbors
//   nn_cnt : Size n, number of selected nearest neighbors for each point
void fsai_exact_knn(
    const int val_type, const int fsai_npt, const int n, const int pt_dim, 
    const void *coord, const int ldc, int *nn_idx, int *nn_cnt
);

// Select k (approximate) nearest neighbors for each point s.t. the 
// indices of neighbors are smaller than the index of the point
// Input parameters:
//   octree     : An initialized octree struct
//   coord0_idx : Size n, for each point in coord, its index in the original input point set
// Other input and output parameters are the same as fsai_exact_knn()
void fsai_octree_fast_knn(
    const int val_type, const int fsai_npt, const int n, const int pt_dim, 
    const void *coord, const int ldc, const int *coord0_idx, octree_p octree, 
    int *nn_idx, int *nn_cnt
);

// Build a Factorized Sparse Approximate Inverse (FSAI) preconditioner for a kernel 
// matrix f^2 * (K(X, X, l) + s * I) + P^T * P, where P is a low rank matrix
// Input parameters:
//   val_type  : Type of the kernel matrix values and point coordinate, 0: double, 1: float
//   krnl_id   : Kernel ID, see kernels/kernels.h
//   param     : Pointer to kernel function parameter array
//   dnoise    : Diagonal noise vector, size of n0, will be ignored if n0 != n1. Can be NULL (== 0)
//   npt       : Number of points in coord
//   pt_dim    : Dimension of each point
//   coord     : Size pt_dim * ldc, row major, each column is a point coordinate
//   ldc       : Leading dimension of coord, >= n
//   fsai_npt  : Maximum number of nonzeros in each row of the FSAI matrix
//   nn_idx    : Size n * fsai_npt, row-major, nearest neighbors indices
//   nn_displs : Size n + 1, start index of each row in nn_idx
//   n1        : Number of columns in P
//   P         : Size n1 * n, col-major, each column is a low rank basis
//   need_grad : If we need to compute gradient matrices (if 0, GdK12 and GdV12 will be ignored)
//   GdK12     : Size n_grad * n1 * n, n_grad col-major n1 * n matrices, for computing dG
//   GdV12     : Size n_grad * n1 * n, n_grad col-major n1 * n matrices, for computing dG
// Output parameter:
//   fp : Pointer to an initialized fsai_precond struct
void fsai_precond_build(
    const int val_type, const int krnl_id, const void *param, const void *dnoise, 
    const int npt, const int pt_dim, const void *coord, const int ldc, 
    const int fsai_npt, const int *nn_idx, const int *nn_displs, 
    const int n1, const void *P, const int need_grad, 
    const void *GdK12, const void *GdV12, fsai_precond_p *fp
);

// Free an initialized fsai_precond struct
void fsai_precond_free(fsai_precond_p *fp);

// Apply the FSAI preconditioner to multiple column vectors
// This interface is designed to be the same as matmul_fptr
// Input parameters:
//   fp  : Pointer to an initialized fsai_precond struct
//   n   : Number of columns in matrices B and C
//   B   : Size ldB * n, col-major dense input matrix
//   ldB : Leading dimension of B, >= np->n
//   ldC : Leading dimension of C, >= np->n
// Output parameter:
//   C : Size ldC * n, col-major dense result matrix, M^{-1} * B
void fsai_precond_apply(const void *fp, const int n, const void *B, const int ldB, void *C, const int ldC);

#ifdef __cplusplus
}
#endif

#endif
