#ifndef __GAUSSIAN_KERNEL_H__
#define __GAUSSIAN_KERNEL_H__

#ifdef __cplusplus
extern "C" {
#endif

// Gaussian kernel K(x, y, l) = exp(-|x - y|^2 / (2 * l^2))
// See common.h for krnl_func definitions of input and output parameters
void gaussian_krnl(
    const int n0, const int ld0, const void *c0,
    const int n1, const int ld1, const void *c1,
    const void *param, const int ldm, void *mat, const int val_type
);

// Gaussian kernel derivative d K(x, y, l) / d l 
// See common.h for krnl_func definitions of input and output parameters
void gaussian_grad(
    const int n0, const int ld0, const void *c0,
    const int n1, const int ld1, const void *c1,
    const void *param, const int ldm, void *mat, const int val_type
);

// Gaussian kernel and its derivative w.r.t. parameter l
// See common.h for krnl_grad_func definitions of input and output parameters
// gaussian_krnl() and gaussian_grad() are wrappers of this function
void gaussian_krnl_grad(
    const int n0, const int ld0, const void *c0,
    const int n1, const int ld1, const void *c1,
    const void *param, const int ldm, const int val_type, 
    const int require_krnl, void *k_mat, 
    const int require_grad, void *dl_mat
);

#ifdef __cplusplus
}
#endif

#endif 
