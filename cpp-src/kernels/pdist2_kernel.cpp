#include "../common.h"
#include "../cpu_linalg.hpp"
#include "pdist2_kernel.h"

template<typename VT>
static void pdist2_krnl(
    const int n0, const int ld0, const VT *c0,
    const int n1, const int ld1, const VT *c1,
    const int dim, const int ldm, VT *mat
)
{
    // 1. mat(:, 1) = sum(c0.^2, 2); 
    //    mat(1, :) = sum(c1.^2, 2)';
    for (int i = 0; i < n0; i++) mat[i] = 0;
    for (int j = 0; j < n1; j++) mat[j * ldm] = 0;
    VT c00 = 0, c10 = 0;
    for (int d = 0; d < dim; d++)
    {
        c00 += c0[d * ld0] * c0[d * ld0];
        for (int i = 1; i < n0; i++)
        {
            VT c0di = c0[d * ld0 + i];
            mat[i] += c0di * c0di;
        }
        c10 += c1[d * ld1] * c1[d * ld1];
        for (int j = 1; j < n1; j++)
        {
            VT c1dj = c1[d * ld1 + j];
            mat[j * ldm] += c1dj * c1dj;
        }
    }

    // 2. d2(i, j) = sum(c0.^2, 2)(i) + sum(c1.^2, 2)(j);
    for (int j = 1; j < n1; j++)
    {
        VT mat_0j = mat[j * ldm];
        VT *mat_j = mat + j * ldm;
        #pragma omp simd
        for (int i = 1; i < n0; i++) mat_j[i] = mat[i] + mat_0j;
    }
    for (int i = 1; i < n0; i++) mat[i] += c10;
    for (int j = 1; j < n1; j++) mat[j * ldm] += c00;
    mat[0] = c00 + c10;

    // 3. d2 = d2 - 2 * c0 * c1';
    VT v_one = 1, v_neg_two = -2;
    xgemm_(
        notrans, trans, &n0, &n1, &dim,
        &v_neg_two, c0, &ld0, c1, &ld1,
        &v_one, mat, &ldm
    );
}

void pdist2_krnl(
    const int n0, const int ld0, const void *c0,
    const int n1, const int ld1, const void *c1,
    const void *param, const int ldm, void *mat, const int val_type
)
{
    if (val_type == VAL_TYPE_DOUBLE)
    {
        const double *c0_d = (const double *) c0;
        const double *c1_d = (const double *) c1;
        const double *param_d = (const double *) param;
        double *mat_d = (double *) mat;
        int dim = (int) param_d[0];
        pdist2_krnl<double>(n0, ld0, c0_d, n1, ld1, c1_d, dim, ldm, mat_d);
    }
    
    if (val_type == VAL_TYPE_FLOAT)
    {
        const float *c0_f = (const float *) c0;
        const float *c1_f = (const float *) c1;
        const float *param_f = (const float *) param;
        float *mat_f = (float *) mat;
        int dim = (int) param_f[0];
        pdist2_krnl<float>(n0, ld0, c0_f, n1, ld1, c1_f, dim, ldm, mat_f);
    }
}
