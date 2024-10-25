#ifndef __PDIST2_KERNEL_H__
#define __PDIST2_KERNEL_H__

#ifdef __cplusplus
extern "C" {
#endif

// Pairwise distance kernel K(x, y) = norm(x - y, 2)
// See common.h for krnl_func definitions of input and output parameters
void pdist2_krnl(
    const int n0, const int ld0, const void *c0,
    const int n1, const int ld1, const void *c1,
    const void *param, const int ldm, void *mat, const int val_type
);

#ifdef __cplusplus
}
#endif

#endif 
