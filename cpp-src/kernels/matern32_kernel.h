#ifndef __MATERN32_KERNEL_H__
#define __MATERN32_KERNEL_H__

#ifdef __cplusplus
extern "C" {
#endif


// Matern 3/2 kernel K(x, y, l) = (1 + sqrt(3) * d / l) * exp(-sqrt(3) * d / l), d = |x - y|
// See common.h for krnl_func definitions of input and output parameters
void matern32_krnl(
    const int n0, const int ld0, const void *c0,
    const int n1, const int ld1, const void *c1,
    const void *param, const int ldm, void *mat, const int val_type
);

// Matern 3/2 kernel derivative d K(x, y, l) / d l 
// See common.h for krnl_func definitions of input and output parameters
void matern32_grad(
    const int n0, const int ld0, const void *c0,
    const int n1, const int ld1, const void *c1,
    const void *param, const int ldm, void *mat, const int val_type
);

// Matern 3/2 kernel and its derivative w.r.t. parameter l
// See common.h for krnl_grad_func definitions of input and output parameters
// matern32_krnl() and matern32_grad() are wrappers of this function
void matern32_krnl_grad(
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
