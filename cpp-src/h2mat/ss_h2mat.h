#ifndef __SS_H2MAT_H__
#define __SS_H2MAT_H__

#include "octree.h"
#include "h2mat_typedef.h"

struct ss_h2mat
{
    void    *param;         // Kernel matrix parameters [dim, l, f, s]
    void    *dnoise;        // Diagonal noise vector
    h2mat_p K_h2mat;        // H2 matrix for K(X, X, l)
    h2mat_p dKdl_h2mat;     // H2 matrix for dK(X, X, l) / dl
};
typedef struct ss_h2mat  ss_h2mat_s;
typedef struct ss_h2mat* ss_h2mat_p;

#ifdef __cplusplus
extern "C" {
#endif

// Initialize an ss_h2mat struct for computing 
// f^2 * K(X, X, l) + s * I + diag(dnoise) and its derivative w.r.t. l, f, and s
// Input parameters:
//   octree  : Point partitioning octree
//   param   : Kernel matrix parameters [dim, l, f, s]
//   dnoise  : Diagonal noise vector, can be NULL (== 0)
//   krnl_id : Kernel ID, see kernels/kernels.h
//   algo    : H2 matrix construction algorithm, 1 for proxy points
//   reltol  : H2 compression relative error tolerance
// Output parameter:
//   *ss_h2mat : An initialized ss_h2mat struct
// Note: algo only supports 1 for now
void ss_h2mat_init(
    octree_p octree, const void *param, const void *dnoise, const int krnl_id, 
    const int algo, void *reltol, ss_h2mat_p *ss_h2mat
);

// Free an initialized ss_h2mat struct
void ss_h2mat_free(ss_h2mat_p *ss_h2mat);

// Compute C = M * B, M is the dense kernel matrix or its derivate matrices, 
// B is a dense input matrix, and C is a dense result matrix
// Input parameters:
//   ss_h2mat : An initialized ss_h2mat struct for K
//   B_ncol   : Number of columns of B
//   B        : Dense input matrix B, col-major, size of ldB * B_ncol
//   ldB      : Leading dimension of B, >= dkmat->ncol
//   ldC      : Leading dimension of C, >= dkmat->nrow
// Output parameters:
//   K_B            : Dense result matrix K * B, col-major, size of dkmat->nrow * B_ncol
//   dKd{l, f, s}_B : Dense result matrix (d K / d {l, f, s}) * B, col-major, size of 
//                    dkmat->nrow * B_ncol, will not be computed if the input pointer is NULL
void ss_h2mat_grad_matmul(
    ss_h2mat_p ss_h2mat, const int B_ncol, void *B, const int ldB, 
    void *K_B, void *dKdl_B, void *dKdf_B, void *dKds_B, const int ldC
);

// Compute K * B only, parameters are the same as ss_h2mat_grad_matmul()
void ss_h2mat_krnl_matmul(
    ss_h2mat_p ss_h2mat, const int B_ncol, void *B, const int ldB, 
    void *K_B, const int ldC
);

#ifdef __cplusplus
}
#endif

#endif
