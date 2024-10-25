#ifndef __DENSE_KERNEL_MATRIX_H__
#define __DENSE_KERNEL_MATRIX_H__

#include "kernels/kernels.h"

// Dense kernel matrix K(X, Y, l, f, s) = f^2 * k(X, Y, l) + s * I + diag(dnoise) and 
// its derivatives w.r.t. hyperparameters f, l, and s
struct dense_krnl_mat
{
    int  nrow, ncol;    // Number of rows and columns of a dense kernel matrix
    int  pt_dim;        // Data point dimension
    int  val_type;      // Data type of coordinate and kernel values, 0 for double, 1 for float
    int  krnl_id;       // Kernel ID, see kernels/kernels.h
    int  same_XY;       // If two input point sets are the same. If not, s * I will be ignored.
    void *X, *Y;        // Coordinate values of X and Y, col-major, size of {nrow, ncol} * dim
    void *param;        // Kernel matrix parameters [dim, l, f, s]
    void *dnoise;       // Diagonal noise vector, size of nrow, will be ignored if same_XY == 0
    void *k_mat;        // Kernel matrix K(X, Y, l), col-major, size of nrow * ncol
    void *dl_mat;       // d K(X, Y, l) / d l, col-major, size of nrow * ncol
};
typedef struct dense_krnl_mat  dense_krnl_mat_s;
typedef struct dense_krnl_mat *dense_krnl_mat_p;

#ifdef __cplusplus
extern "C" {
#endif

// Initialize a dense_krnl_mat struct for computing 
// f^2 * K(X, X, l) + s * I + diag(dnoise) and its derivative w.r.t. l, f, and s
// Input parameters:
//   n{0, 1}  : {nrow, ncol} in dense_krnl_mat
//   ld{0, 1} : Leading dimension of c{0, 1}, >= n{0, 1}
//   c{0, 1}  : Coordinate values of X and Y, col-major, size of ld{0, 1} * dim
//   param    : Kernel matrix parameters [dim, l, f, s]
//   dnoise   : Diagonal noise vector, size of n0, will be ignored if n0 != n1. Can be NULL (== 0)
//   krnl_id  : Kernel ID, see kernels/kernels.h
//   val_type : Data type of coordinate and kernel values, 0 for double, 1 for float
// Output parameter:
//   dkmat : An initialized dense_krnl_mat struct
void dense_krnl_mat_init(
    const int n0, const int ld0, const void *c0, 
    const int n1, const int ld1, const void *c1, 
    const void *param, const void *dnoise, const int krnl_id, 
    const int val_type, dense_krnl_mat_p *dkmat
);

// Free a dense_krnl_mat struct
void dense_krnl_mat_free(dense_krnl_mat_p *dkmat);

// Try to compute K(X, Y, l) and d K(X, Y, l) / d l and store them in dkmat.
// If the system has enough memory to store K(X, Y, l) and d K(X, Y, l) / d l, 
// dkmat->k_mat and dkmat->dl_mat will be populated and the function returns 1.
// Otherwise, the function will return 0.
// Calling dense_krnl_mat_grad_eval() will call this function.
int dense_krnl_mat_populate(dense_krnl_mat_p dkmat);

// Compute C = M * B, M is the dense kernel matrix or its derivate matrices, 
// B is a dense input matrix, and C is a dense result matrix
// Note: if dkmat->k_mat == NULL && dkmat->dl_mat == NULL, these two matrices
// will be generated on-the-fly and will not be stored in dkmat.
// Input parameters:
//   dkmat    : An initialized dense_krnl_mat struct for K
//   B_ncol   : Number of columns of B
//   B        : Dense input matrix B, col-major, size of ldB * B_ncol
//   ldB      : Leading dimension of B, >= dkmat->ncol
//   ldC      : Leading dimension of C, >= dkmat->nrow
// Output parameters:
//   K_B            : Dense result matrix K * B, col-major, size of dkmat->nrow * B_ncol
//   dKd{l, f, s}_B : Dense result matrix (d K / d {l, f, s}) * B, col-major, size of 
//                    dkmat->nrow * B_ncol, will not be computed if the input pointer is NULL
void dense_krnl_mat_grad_matmul(
    dense_krnl_mat_p dkmat, const int B_ncol, void *B, const int ldB, 
    void *K_B, void *dKdl_B, void *dKdf_B, void *dKds_B, const int ldC
);

// Compute K * B only, parameters are the same as dense_krnl_mat_grad_matmul()
void dense_krnl_mat_krnl_matmul(
    dense_krnl_mat_p dkmat, const int B_ncol, void *B, const int ldB, 
    void *K_B, const int ldC
);

// Evaluate K = K(X, Y, l, f, s) and d K / d {l, f, s}
// Warning: only use this for small matrices! This function will 
// calls dense_krnl_mat_populate().
// Input parameter:
//   dkmat : An initialized dense_krnl_mat struct
// Output parameters:
//   K            : Dense kernel matrix K,  col-major, size of dkmat->nrow * dkmat->ncol
//   dKd{l, f, s} : Dense matrix d K / d {l, f, s}, col-major, size of dkmat->nrow * dkmat->ncol
// Note: K and dKd{l, f, s} can be NULL, and the corresponding matrix will not be computed
void dense_krnl_mat_grad_eval(dense_krnl_mat_p dkmat, void *K, void *dKdl, void *dKdf, void *dKds);

#ifdef __cplusplus
}
#endif

#endif
