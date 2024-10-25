#ifndef __CUSTOM_KERNEL_H__
#define __CUSTOM_KERNEL_H__

#ifdef __cplusplus
extern "C" {
#endif

// Custom kernel K(x, y, l). Currently implemented as the Gaussian kernel.
// See common.h for krnl_func definitions of input and output parameters
void custom_krnl(
    const int n0, const int ld0, const void *c0,
    const int n1, const int ld1, const void *c1,
    const void *param, const int ldm, void *mat, const int val_type
);

// Custom kernel K(x, y, l)
// See common.h for krnl_func definitions of input and output parameters
void custom_grad(
    const int n0, const int ld0, const void *c0,
    const int n1, const int ld1, const void *c1,
    const void *param, const int ldm, void *mat, const int val_type
);

// Custom kernel and its derivative w.r.t. parameter l
// See common.h for krnl_grad_func definitions of input and output parameters
// custom_krnl() and custom_grad() are wrappers of this function
void custom_krnl_grad(
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
