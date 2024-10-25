#ifndef __DISPATCH_KERNELS_H__
#define __DISPATCH_KERNELS_H__

#define DISPATCH_KERNELS(VT, KRNL_123D, KRNL_GENERIC) \
do {  \
    const VT *c0_v = (const VT *) c0;  \
    const VT *c1_v = (const VT *) c1;  \
    const VT *param_v = (const VT *) param;  \
    VT *k_mat_v  = (VT *) k_mat;   \
    VT *dl_mat_v = (VT *) dl_mat;  \
    int dim = (int) param_v[0];  \
    VT l_v = param_v[1];  \
    if (require_krnl == 1 && require_grad == 1)  \
    {  \
        if (dim == 1) KRNL_123D<VT, 1, 1, 1>(n0, ld0, c0_v, n1, ld1, c1_v,      l_v, ldm, k_mat_v, dl_mat_v);  \
        if (dim == 2) KRNL_123D<VT, 2, 1, 1>(n0, ld0, c0_v, n1, ld1, c1_v,      l_v, ldm, k_mat_v, dl_mat_v);  \
        if (dim == 3) KRNL_123D<VT, 3, 1, 1>(n0, ld0, c0_v, n1, ld1, c1_v,      l_v, ldm, k_mat_v, dl_mat_v);  \
        if (dim >= 4) KRNL_GENERIC<VT, 1, 1>(n0, ld0, c0_v, n1, ld1, c1_v, dim, l_v, ldm, k_mat_v, dl_mat_v);  \
    }  \
    if (require_krnl == 1 && require_grad == 0)  \
    {  \
        if (dim == 1) KRNL_123D<VT, 1, 1, 0>(n0, ld0, c0_v, n1, ld1, c1_v,      l_v, ldm, k_mat_v, dl_mat_v);  \
        if (dim == 2) KRNL_123D<VT, 2, 1, 0>(n0, ld0, c0_v, n1, ld1, c1_v,      l_v, ldm, k_mat_v, dl_mat_v);  \
        if (dim == 3) KRNL_123D<VT, 3, 1, 0>(n0, ld0, c0_v, n1, ld1, c1_v,      l_v, ldm, k_mat_v, dl_mat_v);  \
        if (dim >= 4) KRNL_GENERIC<VT, 1, 0>(n0, ld0, c0_v, n1, ld1, c1_v, dim, l_v, ldm, k_mat_v, dl_mat_v);  \
    }  \
    if (require_krnl == 0 && require_grad == 1)  \
    {  \
        if (dim == 1) KRNL_123D<VT, 1, 0, 1>(n0, ld0, c0_v, n1, ld1, c1_v,      l_v, ldm, k_mat_v, dl_mat_v);  \
        if (dim == 2) KRNL_123D<VT, 2, 0, 1>(n0, ld0, c0_v, n1, ld1, c1_v,      l_v, ldm, k_mat_v, dl_mat_v);  \
        if (dim == 3) KRNL_123D<VT, 3, 0, 1>(n0, ld0, c0_v, n1, ld1, c1_v,      l_v, ldm, k_mat_v, dl_mat_v);  \
        if (dim >= 4) KRNL_GENERIC<VT, 0, 1>(n0, ld0, c0_v, n1, ld1, c1_v, dim, l_v, ldm, k_mat_v, dl_mat_v);  \
    }  \
} while (0);

#endif
