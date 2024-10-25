#ifndef __BPCG_H__
#define __BPCG_H__

#include "../common.h"

#ifdef __cplusplus
extern "C" {
#endif

// Blocked (multiple RHS) preconditioned conjugate gradient
// Reference: Saad book Algorithm 9.1, formula (6.103)
// Input parameters:
//   m        : Size of the linear system
//   n        : Number of RHS 
//   val_type : Data type of A, invM, B, and X, 0 for double, 1 for float
//   max_iter : Maximum number of iterations to perform
//   *reltol  : Relative tolerance
//   A_mm     : Pointer to a matmul function that computes A * X
//   A        : Pointer to a data structure that represents a matrix A of size m * m
//   invM_mm  : Pointer to a matmul function that computes inv(M) * X, can be NULL
//   invM     : Pointer to a data structure that represents a matrix inv(M) of size m * m, can be NULL
//   B        : Size >= ldB * n, col-major dense input matrix
//   ldB      : Leading dimension of B, >= m
//   X        : Size >= ldX * n, col-major dense initial guess matrix
//   ldX      : Leading dimension of X, >= m
// Output parameters:
//   X     : Size >= ldX * n, col-major dense result matrix
//   iters : Optional, size n, number of iterations for each RHS to converge, can be NULL
void bpcg(
    const int m, const int n, const int val_type, const int max_iter, const void *reltol, 
    matmul_fptr A_mm, const void *A, matmul_fptr invM_mm, const void *invM, 
    const void *B, const int ldB, void *X, const int ldX, int *iters
);

#ifdef __cplusplus
}
#endif

#endif
