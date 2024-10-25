#ifndef __NYS_PRECOND_H__
#define __NYS_PRECOND_H__

struct nys_precond
{
    int  n, k;          // Size of the kernel matrix and the Nystrom approximation rank
    int  val_type;      // Type of the kernel matrix values, 0: double, 1: float
    int  *perm;         // Size n, the i-th point after permutation is the perm[i]-th original point
    void *U;            // Size n * k, col-major
    void *M;            // Size k
    void *eta;          // Size 1, diagonal shift of the kernel matrix
    void *L11;          // Size k * k, col-major, K11 block lower Cholesky factor
    void *K1;           // Size k * n, col-major, K1 block
    void *dK1;          // Size 3 * k * n, col-major, d K1 / d {l, f, s}
    void *logdet;       // Size 1, log(det(M))
    void *gt;           // Size 3, trace of M^{-1} * dM / d {l, f, s}
};
typedef struct nys_precond  nys_precond_s;
typedef struct nys_precond *nys_precond_p;

#ifdef __cplusplus
extern "C" {
#endif

// Build a Nystrom approximation preconditioner for f^2 * (K(coord, coord, l) + s * I)
// Input parameters:
//   val_type  : Type of the kernel matrix values and point coordinate, 0: double, 1: float
//   krnl_id   : Kernel ID, see kernels/kernels.h
//   param     : Pointer to kernel function parameter array, [dim, l, f, s]
//   dnoise    : Diagonal noise vector, size of n0, will be ignored if n0 != n1. Can be NULL (== 0)
//   npt       : Number of points in coord
//   pt_dim    : Dimension of each point
//   coord     : Matrix, size ldc * pt_dim, col-major, each row is a point coordinate
//   ldc       : Leading dimension of coord
//   perm      : Size npt, perm[0 : k-1] are the landmark points
//   k         : Nystrom approximation rank
//   need_grad : If need to compute the gradient matrices dM^{-1} / d {l, f, s}
// Output parameter:
//   np : Pointer to an initialized nys_precond struct
void nys_precond_build(
    const int val_type, const int krnl_id, const void *param, const void *dnoise, 
    const int npt, const int pt_dim, const void *coord, const int ldc, 
    const int *perm, const int k, const int need_grad, nys_precond_p *np
);

// Free an initialized nys_precond struct
void nys_precond_free(nys_precond_p *np);

// Apply the Nystrom preconditioner to multiple column vectors
// This interface is designed to be the same as matmul_fptr
// Input parameters:
//   np  : Pointer to an initialized nys_precond struct
//   n   : Number of columns in matrices B and C
//   B   : Size ldB * n, col-major dense input matrix
//   ldB : Leading dimension of B, >= np->n
//   ldC : Leading dimension of C, >= np->n
// Output parameter:
//   C : Size ldC * n, col-major dense result matrix, M^{-1} * B
void nys_precond_apply(const void *np, const int n, const void *B, const int ldB, void *C, const int ldC);

// Compute Y := d M^{-1} / d {l, f, s} * X, M is the Nystrom precond matrix
// Input parameters:
//   ap        : Pointer to an initialized nys_precond struct
//   n         : Number of columns in matrices X
//   X         : Size ldX * n, col-major dense input matrix
//   ldX       : Leading dimension of X, >= np->n
//   ldY       : Leading dimension of Y, >= np->n
//   skip_perm : If skip the permutation of X and Y, only used by nys_precond_grad_trace()
// Output parameter:
//   Y : Size ldY * n * 3, col-major dense result matrix
void nys_precond_dapply(nys_precond_p np, const int n, const void *X, const int ldX, void *Y, const int ldY, const int skip_perm);

// Compute the trace of M^{-1} * dM / d {l, f, s}, store the result in np->gt
void nys_precond_grad_trace(nys_precond_p np);

#ifdef __cplusplus
}
#endif

#endif
