#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <omp.h>

#include "mfom.h"
#include "../cpu_linalg.hpp"
#include "../utils.h"

template<typename VT>
static void mfom(
    const int m, const int n, const int n_iter, 
    const matmul_fptr A_mm, const void *A, 
    const VT *B, const int ldB, VT *X, const int ldX, VT *T
)
{
    const int n_iter1 = n_iter + 1;
    VT  *V    = (VT *)  malloc(sizeof(VT)  * m * n_iter1 * n);
    VT  *H    = (VT *)  malloc(sizeof(VT)  * n_iter * n_iter * n);
    VT  *vb   = (VT *)  malloc(sizeof(VT)  * n);
    VT  *vt   = (VT *)  malloc(sizeof(VT)  * n);
    VT  *vp   = (VT *)  malloc(sizeof(VT)  * n_iter);
    VT  *e1   = (VT *)  malloc(sizeof(VT)  * n_iter);
    int *ipiv = (int *) malloc(sizeof(int) * n_iter);
    #pragma omp parallel for schedule(static)
    for (int j = 0; j < n_iter1 * n; j++)
        memset(V + j * m, 0, sizeof(VT) * m);
    memset(H, 0, sizeof(VT) * n_iter * n_iter * n);

    // iidx = 1 : n_iter1 : ((n - 1) * n_iter1 + 1);
    // V(:, iidx) = B - A_mm(X0);
    int ldV = m, ldVmm = m * n_iter1;
    A_mm(A, n, X, ldX, V, ldVmm);

    // vb(ir) = norm(V(:, iidx(ir)), 2);
    // V(:, iidx) = V(:, iidx) ./ vb;
    const int i_one = 1;
    #pragma omp parallel for schedule(static)
    for (int ir = 0; ir < n; ir++)
    {
        VT *V_ir = V + (ir * n_iter1) * ldV;
        const VT *B_ir = B + ir * ldB;
        #pragma omp simd
        for (int i = 0; i < m; i++) V_ir[i] = B_ir[i] - V_ir[i];
        vb[ir] = xnrm2_(&m, V_ir, &i_one);
        VT inv_b = 1.0 / vb[ir];
        #pragma omp simd
        for (int i = 0; i < m; i++) V_ir[i] *= inv_b;
    }
    
    VT v_neg_one = -1.0, v_zero = 0.0, v_one = 1.0;
    for (int i = 0; i < n_iter; i++)
    {
        // iidx0 = i : n_iter1 : ((nrhs - 1) * n_iter1 + i);
        // iidx1 = iidx0 + 1;
        // V(:, iidx1) = A_mm(V(:, iidx0));
        A_mm(A, n, V + i * m, ldVmm, V + (i + 1) * m, ldVmm);

        // CGS twice to orthogonalize V(:, iidx1) against V(:, 1 : iidx0)
        for (int k = 0; k < 2; k++)
        {
            for (int ir = 0; ir < n; ir++)
            {
                // base_idx = (ir - 1) * n_iter1;
                // ir_i0 = (base_idx + 1) : (base_idx + i);
                // ir_i1 = iidx1(ir);  % == base_idx + i + 1
                int base_idx = ir * n_iter1;
                int i1 = i + 1;
                VT *V_ir_i0 = V + base_idx * ldV;
                VT *V_ir_i1 = V + (base_idx + i + 1) * ldV;
                // vp = V(:, ir_i0)' * V(:, ir_i1);
                xgemv_(
                    trans, &m, &i1, &v_one, V_ir_i0, &ldV,
                    V_ir_i1, &i_one, &v_zero, vp, &i_one
                );
                // V(:, ir_i1) = V(:, ir_i1) - V(:, ir_i0) * vp;
                xgemv_(
                    notrans, &m, &i1, &v_neg_one, V_ir_i0, &ldV,
                    vp, &i_one, &v_one, V_ir_i1, &i_one
                );

                // H{ir}(1 : i, i) = H{ir}(1 : i, i) + vp;
                VT *H_ir_i = H + ir * n_iter * n_iter + i * n_iter;
                #pragma omp simd
                for (int j = 0; j <= i; j++) H_ir_i[j] += vp[j];
            }  // End of ir loop
        }  // End of k loop

        // vt(ir) = norm(V(:, iidx1(ir)), 2);
        // V(:, iidx1) = V(:, iidx1) ./ vt;
        #pragma omp parallel for schedule(static)
        for (int ir = 0; ir < n; ir++)
        {
            VT *V_ir = V + (ir * n_iter1 + i + 1) * ldV;
            vt[ir] = xnrm2_(&m, V_ir, &i_one);
            VT inv_t = 1.0 / vt[ir];
            #pragma omp simd
            for (int i = 0; i < m; i++) V_ir[i] *= inv_t;
        }

        // H{ir}(i+1, i) = vt(ir);
        if (i < n_iter - 1)
        {
            for (int ir = 0; ir < n; ir++)
            {
                VT *H_ir_i = H + ir * n_iter * n_iter + i * n_iter;
                H_ir_i[i + 1] = vt[ir];
            }
        }
    }  // End of i loop

    int info = 0;
    for (int ir = 0; ir < n; ir++)
    {
        // T{ir} = tril(H{ir}, 1);
        VT *H_ir = H + ir * n_iter * n_iter;
        VT *T_ir = T + ir * n_iter * n_iter;
        for (int i = 0; i < n_iter; i++)
        {
            int idx0 = i * n_iter + i - 1;
            int idx1 = i * n_iter + i;
            int idx2 = i * n_iter + i + 1;
            if (i > 0) T_ir[idx0] = H_ir[idx0];
            T_ir[idx1] = H_ir[idx1];
            if (i < n_iter - 1) T_ir[idx2] = H_ir[idx2];
        }

        // y = Hm \ (vb(ir) .* e1);
        e1[0] = vb[ir];
        for (int j = 1; j < n_iter; j++) e1[j] = 0.0;
        // idx_ir = ((ir - 1) * n_iter1 + 1) : (ir * n_iter1 - 1);
        // X(:, ir) = X0(:, ir) + V(:, idx_ir) * y;
        VT *X_ir = X + ir * ldX;
        VT *V_ir = V + ir * n_iter1 * ldV;
        xgesv_(&n_iter, &i_one, H_ir, &n_iter, ipiv, e1, &n_iter, &info);
        ASSERT_PRINTF(info == 0, "xGESV_ failed with info = %d\n", info);
        xgemv_(
            notrans, &m, &n_iter, &v_one, V_ir, &ldV,
            e1, &i_one, &v_one, X_ir, &i_one
        );
    }

    free(V);
    free(H);
    free(vb);
    free(vt);
    free(e1);
    free(ipiv);
}

// Modified multiple RHS full orthogonalization method (FOM)
void mfom(
    const int m, const int n, const int n_iter, const int val_type, 
    matmul_fptr A_mm, const void *A, 
    const void *B, const int ldB, void *X, const int ldX, void *T
)
{
    if (val_type == VAL_TYPE_DOUBLE)
    {
        mfom<double>(
            m, n, n_iter, A_mm, A, 
            (double *) B, ldB, (double *) X, ldX, (double *) T
        );
    }
    if (val_type == VAL_TYPE_FLOAT)
    {
        mfom<float>(
            m, n, n_iter, A_mm, A, 
            (float *) B, ldB, (float *) X, ldX, (float *) T
        );
    }
}