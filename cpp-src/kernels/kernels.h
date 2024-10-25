#ifndef __KERNELS_H__
#define __KERNELS_H__

#include "../common.h"
#include "pdist2_kernel.h"
#include "gaussian_kernel.h"
#include "matern32_kernel.h"
#include "matern52_kernel.h"
#include "custom_kernel.h"

typedef enum kernel_id
{
    KERNEL_ID_PDIST2   = 0,     // Squared distance kernel
    KERNEL_ID_GAUSSIAN = 1,     // Gaussian kernel
    KERNEL_ID_MATERN32 = 2,     // Matern 3/2 kernel
    KERNEL_ID_MATERN52 = 3,     // Matern 5/2 kernel
    KERNEL_ID_CUSTOM   = 99,    // Custom kernel
} kernel_id_t;

#ifdef __cplusplus
extern "C" {
#endif

// Get the krnl_func and/or krnl_grad_func of a kernel by its ID
// Input parameter:
//   krnl_id : Kernel ID, see kernel_id_t
// Output parameters:
//   *krnl      : krnl_func of the kernel, NULL if not required
//   *gkrnl     : krnl_func of the derivative of the kernel w.r.t. l, NULL if not required
//   *krnl_grad : krnl_grad_func of the kernel, NULL if not required
// Note: KERNEL_ID_PDIST2 has no krnl_grad_func
void get_krnl_grad_func(const int krnl_id, krnl_func *krnl, krnl_func *gkrnl, krnl_grad_func *krnl_grad);

// Computing a krnl_func or krnl_grad_func using multiple threads
// Input and output parameters are the same as krnl_func and krnl_grad_func,
// with an extra parameter n_thread specifying the number of threads to use.
// If n_thread <= 0, all available threads will be used.
void krnl_func_omp(
    const int n0, const int ld0, const void *c0,
    const int n1, const int ld1, const void *c1,
    krnl_func krnl, const void *param, const int ldm, void *mat, 
    const int val_type, const int n_thread
);
void krnl_grad_func_omp(
    const int n0, const int ld0, const void *c0,
    const int n1, const int ld1, const void *c1,
    krnl_grad_func krnl_grad, const void *param, const int ldm, 
    const int require_krnl, void *k_mat, 
    const int require_grad, void *dl_mat, 
    const int val_type, const int n_thread
);

// Compute K * B and (d K / d l) * B using multiple threads, the 
// kernel matrix will be populated on-the-fly and discarded after use.
// Parameters in the first three rows are the same as krnl_func and krnl_grad_func,
// with some extra parameters specifying the input matrix B and output matrix C.
// Input parameters:
//   nvec     : Number of columns of B and C
//   B        : Input matrix B (for K and dK/dl), size ldB * nvec, col-major
//   ldB      : Leading dimension of B, >= n1
//   ldC      : Leading dimension of C, >= n0
//   n_thread : Number of threads to use, if <= 0, all available threads will be used
// Output parameter:
//   (krnl_, grad_)C : Output matrix C (for K and dK/dl), size ldC * nvec, col-major.
//                     If a pointer is NULL, the corresponding matrix will not be computed.
void krnl_matmul_omp(
    const int n0, const int ld0, const void *c0,
    const int n1, const int ld1, const void *c1,
    krnl_func krnl, const void *param, const int val_type, 
    const void *B, const int ldB, const int nvec, 
    void *C, const int ldC, const int n_thread
);
void krnl_grad_matmul_omp(
    const int n0, const int ld0, const void *c0,
    const int n1, const int ld1, const void *c1,
    krnl_grad_func krnl_grad, const void *param, const int val_type, 
    const void *B, const int ldB, const int nvec, 
    void *krnl_C, void *grad_C, const int ldC, const int n_thread
);

#ifdef __cplusplus
}
#endif

#endif
