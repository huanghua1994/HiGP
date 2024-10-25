#ifndef __H2MAT_MATMUL_H__
#define __H2MAT_MATMUL_H__

#include "h2mat_typedef.h"

#ifdef __cplusplus
extern "C" {
#endif

// Compute Y := K(X, X, l) * X
// Input parameters:
//   h2mat : Constrcuted H2 matrix for K(X, X, l)
//   n     : Number of columns of X and Y
//   X     : Size ldX * n, col-major, input matrix
//   ldX   : Leading dimension of X, >= h2mat->octree->npt
//   ldY   : Leading dimension of Y, >= h2mat->octree->npt
// Output parameter:
//   Y : Size ldY * n, col-major, output matrix
// Note: X and Y are of the same data type as h2mat->octree->val_type
void h2mat_matmul(
    h2mat_p h2mat, const int n, const void *X, const int ldX, 
    void *Y, const int ldY
);

#ifdef __cplusplus
}
#endif

#endif
