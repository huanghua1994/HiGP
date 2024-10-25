#include <cstdio>
#include <cmath>
#include <type_traits>
#include <omp.h>

#include "../common.h"
#include "matern52_kernel.h"
#include "pdist2_kernel.h"
#include "dispatch_kernels.h"

// Denote d = |x - y|, Matern 5/2 kernel:
// K(d, l) = (1 + sqrt(5) * d / l + 5 * d^2 / (3 * l^2)) * exp(-sqrt(5) * d / l)
// d K(d, l) / d l = 5 * (d^2 * l + sqrt(5) * d^3) / (3 * l^4) * exp(-sqrt(5) * d / l)

#ifndef M_SQRT5
#define M_SQRT5 2.2360679774997896964
#endif

template<typename VT, int dim, int require_krnl, int require_grad>
static void matern52_123d(
    const int n0, const int ld0, const VT *c0,
    const int n1, const int ld1, const VT *c1,
    const VT l, const int ldm, VT *k_mat, VT *dl_mat
)
{
    const VT *x0 = c0, *x1 = c1, *y0 = NULL, *y1 = NULL, *z0 = NULL, *z1 = NULL;
    VT t0 = -M_SQRT5 / l, _5o3l4 = 5.0 / (3.0 * l * l * l * l);
    if (dim >= 2)
    {
        y0 = c0 + ld0;
        y1 = c1 + ld1;
    }
    if (dim >= 3)
    {
        z0 = c0 + ld0 * 2;
        z1 = c1 + ld1 * 2;
    }
    for (int j = 0; j < n1; j++)
    {
        VT x1j = x1[j], y1j = 0, z1j = 0;
        VT *k_mat_j = NULL, *dl_mat_j = NULL;
        if (dim >= 2) y1j = y1[j];
        if (dim >= 3) z1j = z1[j];
        if (require_krnl == 1) k_mat_j  = k_mat  + ldm * j;
        if (require_grad == 1) dl_mat_j = dl_mat + ldm * j;
        #pragma omp simd
        for (int i = 0; i < n0; i++)
        {
            VT dx, dy, dz, d2, d, t1, t2;
            dx = x0[i] - x1j;
            d2 = dx * dx;
            if (dim >= 2)
            {
                dy = y0[i] - y1j;
                d2 += dy * dy;
            }
            if (dim >= 3)
            {
                dz = z0[i] - z1j;
                d2 += dz * dz;
            }
            d = std::sqrt(d2);
            t1 = t0 * d;  // -sqrt(5) * d / l
            t2 = std::exp(t1);
            if (require_krnl == 1) k_mat_j[i]  = (1.0 - t1 + t1 * t1 / 3.0) * t2;
            if (require_grad == 1) dl_mat_j[i] = _5o3l4 * (d2 * l + M_SQRT5 * d2 * d) * t2;
        }
    }
}

template<typename VT, int require_krnl, int require_grad>
static void matern52_generic(
    const int n0, const int ld0, const VT *c0,
    const int n1, const int ld1, const VT *c1,
    const int dim, const VT l, const int ldm, VT *k_mat, VT *dl_mat
)
{
    VT t0 = -M_SQRT5 / l, _5o3l4 = 5.0 / (3.0 * l * l * l * l);

    VT *dist2_mat = k_mat;
    if (require_krnl == 0) dist2_mat = dl_mat;

    VT dim_v = (VT) dim;
    int val_type = (std::is_same<VT, double>::value) ? VAL_TYPE_DOUBLE : VAL_TYPE_FLOAT;
    pdist2_krnl(
        n0, ld0, (const void *) c0, n1, ld1, (const void *) c1, 
        (const void *) &dim_v, ldm, (void *) dist2_mat, val_type
    );
    
    for (int j = 0; j < n1; j++)
    {
        VT *k_mat_j = NULL, *dl_mat_j = NULL, *dist2_mat_j;
        if (require_krnl == 1) k_mat_j  = k_mat  + ldm * j;
        if (require_grad == 1) dl_mat_j = dl_mat + ldm * j;
        dist2_mat_j = dist2_mat + ldm * j;
        #pragma omp simd
        for (int i = 0; i < n0; i++)
        {
            VT d2 = dist2_mat_j[i];
            VT d = std::sqrt(d2);
            VT t1 = t0 * d;  // -sqrt(5) * d / l
            VT t2 = std::exp(t1);
            if (require_krnl == 1) k_mat_j[i]  = (1.0 - t1 + t1 * t1 / 3.0) * t2;
            if (require_grad == 1) dl_mat_j[i] = _5o3l4 * (d2 * l + M_SQRT5 * d2 * d) * t2;
        }
    }
}

void matern52_krnl_grad(
    const int n0, const int ld0, const void *c0,
    const int n1, const int ld1, const void *c1,
    const void *param, const int ldm, const int val_type, 
    const int require_krnl, void *k_mat, 
    const int require_grad, void *dl_mat
)
{
    if (require_krnl == 0 && require_grad == 0)
    {
        printf("What do you want by setting both require_krnl and require_grad to 0?\n");
        return;
    }
    if (val_type == VAL_TYPE_DOUBLE) DISPATCH_KERNELS(double, matern52_123d, matern52_generic);
    if (val_type == VAL_TYPE_FLOAT)  DISPATCH_KERNELS(float,  matern52_123d, matern52_generic);
}

void matern52_krnl(
    const int n0, const int ld0, const void *c0,
    const int n1, const int ld1, const void *c1,
    const void *param, const int ldm, void *mat, const int val_type
)
{
    matern52_krnl_grad(
        n0, ld0, c0, n1, ld1, c1, 
        param, ldm, val_type, 1, mat, 0, NULL
    );
}

void matern52_grad(
    const int n0, const int ld0, const void *c0,
    const int n1, const int ld1, const void *c1,
    const void *param, const int ldm, void *mat, const int val_type
)
{
    matern52_krnl_grad(
        n0, ld0, c0, n1, ld1, c1, 
        param, ldm, val_type, 0, NULL, 1, mat
    );
}