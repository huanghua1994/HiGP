#include <cstdio>
#include <cmath>
#include <type_traits>
#include <omp.h>

#include "../common.h"
#include "custom_kernel.h"
#include "pdist2_kernel.h"
#include "dispatch_kernels.h"

/* =============== Notes for implementing a custom kernel =============== 
 * This is the file you need to modify for implementing a custom kernel.
 * You need to modify two functions in this file: custom_123d() for 
 * computing the kernel matrix for 1D, 2D, and 3D data points, and 
 * custom_generic() for computing the kernel matrix for any dimension
 * data points. 
 *
 * The arguments of custom_123d() and custom_generic() are listed as below:
 * Input arguments:
 *   n0  : Number of data points in the first data set
 *   ld0 : Leading dimension (>= n0) of the c0 matrix
 *   c0  : Size ld0-by-dim, column-major. Each row stores one data point
 *   n1  : Number of data points in the second data set
 *   ld1 : Leading dimension (>= n1) of the c1 matrix
 *   c1  : Size ld1-by-dim, column-major. Each row stores one data point
 *   dim : Dimension of data points
 *   l   : Length-scale parameter
 *   ldm : Leading dimension (>= n0) of the k_mat and dl_mat
 * Output arguments:
 *   k_mat  : Size ldm-by-n1, column-major, kernel matrix
 *   dl_mat : Size ldm-by-n1, column-major, derivative of kernel matrix w.r.t. length-scale
 * 
 * Do not change the arguments of custom_123d() and custom_generic().
 * You only need to modify the computations inside these functions.
 * Currently, these functions first compute the pairwise squared distance
 * (d2 in the code) of data points, then compute the kernel function using
 * d2 and l (length scale).
*/ 

// Custom kernel, currently implemented as the Gaussian kernel:
// K(d, l) = exp(-d^2 / (2 * l^2))
// d K(d, l) / d l = (d^2 / l^3) * exp(-d^2 / (2 * l^2))

template<typename VT, int dim, int require_krnl, int require_grad>
static void custom_123d(
    const int n0, const int ld0, const VT *c0,
    const int n1, const int ld1, const VT *c1,
    const VT l, const int ldm, VT *k_mat, VT *dl_mat
)
{
    const VT *x0 = c0, *x1 = c1, *y0 = NULL, *y1 = NULL, *z0 = NULL, *z1 = NULL;
    VT neg_inv_2l2 = -0.5 / (l * l), inv_l3 = 1.0 / (l * l * l);
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
            // Compute the squared distance and store it in d2
            VT dx, dy = 0, dz = 0, d2;
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
            // Compute the kernel function based on d2
            VT k_ij = std::exp(d2 * neg_inv_2l2);
            if (require_krnl == 1) k_mat_j[i]  = k_ij;
            if (require_grad == 1) dl_mat_j[i] = d2 * k_ij * inv_l3;
        }
    }
}

template<typename VT, int require_krnl, int require_grad>
static void custom_generic(
    const int n0, const int ld0, const VT *c0,
    const int n1, const int ld1, const VT *c1,
    const int dim, const VT l, const int ldm, VT *k_mat, VT *dl_mat
)
{
    VT neg_inv_2l2 = -0.5 / (l * l), inv_l3 = 1.0 / (l * l * l);

    VT *dist2_mat = k_mat;
    if (require_krnl == 0) dist2_mat = dl_mat;

    VT dim_v = (VT) dim;
    int val_type = (std::is_same<VT, double>::value) ? VAL_TYPE_DOUBLE : VAL_TYPE_FLOAT;

    // Compute the pairwise squared distance and store it in dist2_mat
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
            // Compute the kernel function based on d2
            VT k_ij = std::exp(d2 * neg_inv_2l2);
            if (require_krnl == 1) k_mat_j[i]  = k_ij;
            if (require_grad == 1) dl_mat_j[i] = d2 * k_ij * inv_l3;
        }
    }
}

void custom_krnl_grad(
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
    if (val_type == VAL_TYPE_DOUBLE) DISPATCH_KERNELS(double, custom_123d, custom_generic);
    if (val_type == VAL_TYPE_FLOAT)  DISPATCH_KERNELS(float,  custom_123d, custom_generic);
}

void custom_krnl(
    const int n0, const int ld0, const void *c0,
    const int n1, const int ld1, const void *c1,
    const void *param, const int ldm, void *mat, const int val_type
)
{
    custom_krnl_grad(
        n0, ld0, c0, n1, ld1, c1, 
        param, ldm, val_type, 1, mat, 0, NULL
    );
}

void custom_grad(
    const int n0, const int ld0, const void *c0,
    const int n1, const int ld1, const void *c1,
    const void *param, const int ldm, void *mat, const int val_type
)
{
    custom_krnl_grad(
        n0, ld0, c0, n1, ld1, c1, 
        param, ldm, val_type, 0, NULL, 1, mat
    );
}
