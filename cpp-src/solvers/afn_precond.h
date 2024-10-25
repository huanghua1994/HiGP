#ifndef __AFN_PRECOND_H__
#define __AFN_PRECOND_H__

#include "../common.h"
#include "../h2mat/octree.h"
#include "csr_mat.h"
#include "nys_precond.h"
#include "fsai_precond.h"

struct afn_precond 
{
    int  n;             // Size of the kernel matrix, == number of points
    int  n1;            // Size of K11 block (== global low-rank approximation rank)
    int  n2;            // Size of K22 block (== n - n1)
    int  fsai_npt;      // Maximum number of nonzeros in each row of the FSAI matrix
    int  val_type;      // Type of the kernel matrix values, 0: double, 1: float
    int  est_rank;      // Estimated rank of the kernel matrix (without diagonal shift)
    int  is_nys;        // If the kernel matrix is low-rank and we fall back to Nystrom only
    int  *perm;         // Size n, the i-th point after permutation is the perm[i]-th original point
    void *L11;          // Size n1 * n1, col-major, K11 block lower Cholesky factor
    void *K12;          // Size n1 * n2, col-major, K12 block
    void *dL11;         // Size 3 * n1 * n1, col-major, d L11 / d {l, f, s}
    void *dK12;         // Size 3 * n1 * n2, col-major, d K12 / d {l, f, s}
    void *logdet;       // Size 1, log(det(M))
    void *gt;           // Size 3, trace of M^{-1} * dM / d {l, f, s}
    nys_precond_p  np;  // Nystrom preconditioner if the kernel matrix is low-rank
    fsai_precond_p fp;  // FSAI preconditioner for K22
};
typedef struct afn_precond  afn_precond_s;
typedef struct afn_precond *afn_precond_p;

#ifdef __cplusplus
extern "C" {
#endif

// Build an AFN preconditioner for f^2 * (K(X, X, l) + s * I)
// Input parameters:
//   val_type  : Type of the kernel matrix values and point coordinate, 0: double, 1: float
//   krnl_id   : Kernel ID, see kernels/kernels.h
//   param     : Pointer to kernel function parameter array, [dim, l, f, s]
//   dnoise    : Diagonal noise vector, size of n0, will be ignored if n0 != n1. Can be NULL (== 0)
//   npt       : Number of points in coord
//   pt_dim    : Dimension of each point
//   coord     : Matrix, size npt * pt_dim, col-major, each row is a point coordinate
//   ldc       : Leading dimension of coord
//   npt_s     : Number of points to sample for rank estimation, usually <= 500. 
//               If <= 0, will use -npt_s as ap->est_rank.
//   glr_rank  : Use a global low-rank approximation (Nystrom) with no larger than this rank
//   fsai_npt  : Maximum number of nonzeros in each row of the FSAI matrix
//   octree    : Pointer to an initialized octree struct for fast KNN search; if NULL, use naive search
//   need_grad : If need to compute the gradient matrices dM^{-1} / d {l, f, s}
// Output parameter:
//   ap : Pointer to a constructed afn_precond struct
void afn_precond_build(
    const int val_type, const int krnl_id, const void *param, const void *dnoise, 
    const int npt, const int pt_dim, const void *coord, const int ldc, 
    const int npt_s, const int glr_rank, const int fsai_npt, 
    octree_p octree, const int need_grad, afn_precond_p *ap
);

// Free an initialized afn_precond struct
void afn_precond_free(afn_precond_p *ap);

// Apply the AFN preconditioner to multiple column vectors
// This interface is designed to be the same as matmul_fptr
// Input parameters:
//   ap  : Pointer to an initialized afn_precond struct
//   n   : Number of columns in matrices B and C
//   B   : Size ldB * n, col-major dense input matrix
//   ldB : Leading dimension of B, >= ap->npt
//   ldC : Leading dimension of C, >= ap->npt
// Output parameter:
//   C : Size ldC * n, col-major dense result matrix, inv(M) * B
void afn_precond_apply(const void *ap, const int n, const void *B, const int ldB, void *C, const int ldC);

// Compute Y := d M^{-1} / d {l, f, s} * X, M is the AFN precond matrix
// Input parameters:
//   ap   : Pointer to an initialized afn_precond struct
//   n    : Number of columns in matrices X
//   X    : Size ldX * n, col-major dense input matrix
//   ldX  : Leading dimension of X, >= ap->npt
//   ldY  : Leading dimension of Y, >= ap->npt
// Output parameter:
//   Y : Size ldY * n * 3, col-major dense result matrix
void afn_precond_dapply(afn_precond_p ap, const int n, const void *X, const int ldX, void *Y, const int ldY);

// Compute the trace of the AFN gradient matrices dM^{-1}/dx
// Input parameter:
//   ap : Pointer to an initialized afn_precond struct
// Output parameter:
//   ap->gt : Size 3, d M^{-1} / d {l, f, s} trace
void afn_precond_grad_trace(afn_precond_p ap);

#ifdef __cplusplus
}
#endif

#endif
