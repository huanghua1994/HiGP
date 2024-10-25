#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <limits>
#include <omp.h>

#include "mpcg.h"
#include "../utils.h"
#include "../cpu_linalg.hpp"

// sum(A .* B, 1) (sum each column of A .* B)
template<typename VT>
static void dot_mul_sum(
    const int nrow, const int ncol, const VT *A, const int ldA, 
    const VT *B, const int ldB, VT *s, const int use_omp
)
{
    int ione = 1;
    #pragma omp parallel for if(use_omp == 1) schedule(static)
    for (int j = 0; j < ncol; j++)
    {
        VT *Aj = (VT *) A + j * ldA;
        VT *Bj = (VT *) B + j * ldB;
        s[j] = xdot_(&nrow, Aj, &ione, Bj, &ione);
    }
}

template<typename VT>
static void mpcg(
    const int m, const int n, const int n_iter, 
    const matmul_fptr A_mm, const void *A, const matmul_fptr invM_mm, const void *invM, 
    const VT *B, const int ldB, VT *X, const int ldX, VT *T
)
{
    const size_t val_bytes = sizeof(VT);
    const int ldR = m, ldZ = m, ldP = m, ldS = m, use_omp = 1;
    const VT eps = std::numeric_limits<VT>::epsilon();
    VT *R = (VT *) malloc(sizeof(VT) * m * n);  // Residual vectors
    VT *Z = (VT *) malloc(sizeof(VT) * m * n);  // Preconditioned residual vectors
    VT *P = (VT *) malloc(sizeof(VT) * m * n);  // Search direction vectors
    VT *S = (VT *) malloc(sizeof(VT) * m * n);  // A * P
    VT *v = (VT *) malloc(sizeof(VT) * n * 5);  // Short vectors
    VT *va  = v, *vb = v + n;                   // alpha and beta for each rhs
    VT *va0 = v + n * 2, *vb0 = v + n * 3;      // alpha ane beta in the previous iteration
    VT *vr0 = v + n * 4;      
    
    // Unnecessary initialization, just to silence GCC's Wmaybe-uninitialized
    #pragma omp parallel for if(use_omp == 1) schedule(static)
    for (int i = 0; i < m * n; i++) R[i] = 0.0;

    // R = B - A_mm(X);
    A_mm(A, n, X, ldX, R, ldR);
    for (int j = 0; j < n; j++)
    {
        VT *Rj = R + j * ldR;
        const VT *Bj = B + j * ldB;
        #pragma omp simd
        for (int i = 0; i < m; i++) Rj[i] = Bj[i] - Rj[i];
    }
    // Z = invM_mm(R);
    if (invM_mm != NULL) invM_mm(invM, n, R, ldR, Z, ldZ);
    else copy_matrix(val_bytes, n, m, R, ldR, Z, ldZ, use_omp);  // copy_matrix uses row-major
    // P = Z;
    copy_matrix(val_bytes, n, m, Z, ldZ, P, ldP, use_omp);
    // Initialize all T matrices as 0
    memset(T,   0, sizeof(VT) * n * n_iter * n_iter);
    memset(va0, 0, sizeof(VT) * n);
    memset(vb0, 0, sizeof(VT) * n);

    int *conv = (int *) malloc(sizeof(int) * n);  // Record the converged iteration number for each rhs
    for (int j = 0; j < n; j++) conv[j] = n_iter - 1;

    for (int i = 0; i < n_iter; i++)
    {
        // va0 = va;  vb0 = vb;
        memcpy(va0, va, sizeof(VT) * n);
        memcpy(vb0, vb, sizeof(VT) * n);
        // vr0 = sum(R .* Z)
        dot_mul_sum<VT>(m, n, R, ldR, Z, ldZ, vr0, use_omp);

        // S = A_mm(P);
        A_mm(A, n, P, ldP, S, ldS);
        // va = vr0 ./ sum(S .* P);
        dot_mul_sum<VT>(m, n, S, ldS, P, ldP, va, use_omp);
        for (int j = 0; j < n; j++)
        {
            // If PCG is converged, we simply set va[j] = 1.0 to keep the T{i} computing running
            if (std::abs(vr0[j]) <= eps * 10) va[j] = 1.0;
            else va[j] = vr0[j] / va[j];
        }
        // X = X + va .* P;
        // R = R - va .* S;
        #pragma omp parallel for if(use_omp == 1) schedule(static)
        for (int j = 0; j < n; j++)
        {
            VT *Xj = X + j * ldX;
            VT *Rj = R + j * ldR;
            VT *Pj = P + j * ldP;
            VT *Sj = S + j * ldS;
            VT vaj = va[j];
            #pragma omp simd
            for (int l = 0; l < m; l++)
            {
                Xj[l] += vaj * Pj[l];
                Rj[l] -= vaj * Sj[l];
            }
        }

        if (i < n_iter - 1)
        {
            // Z = P_mm(R);
            if (invM_mm != NULL) invM_mm(invM, n, R, ldR, Z, ldZ);
            else copy_matrix(val_bytes, n, m, R, ldR, Z, ldZ, use_omp);
            // vb = sum(R .* Z) ./ vr0;
            dot_mul_sum<VT>(m, n, R, ldR, Z, ldZ, vb, use_omp);
            for (int j = 0; j < n; j++)
            {
                if (std::abs(vb[j]) <= eps * 10) 
                {
                    // PCG converged, mark it
                    vb[j] = 0.0;
                    conv[j] = (i < conv[j]) ? i : conv[j];
                } else {
                    vb[j] = vb[j] / vr0[j];
                }
            }
            // P = Z + vb .* P;
            #pragma omp parallel for if(use_omp == 1) schedule(static)
            for (int j = 0; j < n; j++)
            {
                VT *Pj = P + j * ldP;
                VT *Zj = Z + j * ldZ;
                VT vbj = vb[j];
                #pragma omp simd
                for (int l = 0; l < m; l++) Pj[l] = Zj[l] + vbj * Pj[l];
            }
        }  // End of "if (i < n_iter - 1)"

        for (int j = 0; j < n; j++)
        {
            VT *Tj = T + j * n_iter * n_iter;
            #define Tj(i, j) Tj[(i) + (j) * n_iter]
            Tj(i, i) = 1.0 / va[j];
            if (i > 0)
            {
                Tj(i, i)  += vb0[j] / va0[j];
                Tj(i-1, i) = std::sqrt(vb0[j]) / va0[j];
                Tj(i, i-1) = std::sqrt(vb0[j]) / va0[j];
            }
            #undef Tj
        }  // End of j loop
    }  // End of i loop

    // Keep only the converged part of each Ti
    for (int j = 0; j < n; j++)
    {
        VT *Tj = T + j * n_iter * n_iter;
        #define Tj(i, j) Tj[(i) + (j) * n_iter]
        for (int i = conv[j] + 1; i < n_iter; i++)
        {
            Tj(i, i)   = 0.0;
            Tj(i-1, i) = 0.0;
            Tj(i, i-1) = 0.0;
        }
        #undef Tj
    }

    free(R);
    free(Z);
    free(P);
    free(S);
    free(v);
    free(conv);
}

// Modified multiple RHS preconditioned conjugate gradient
void mpcg(
    const int m, const int n, const int n_iter, const int val_type, 
    const matmul_fptr A_mm, const void *A, const matmul_fptr invM_mm, const void *invM, 
    const void *B, const int ldB, void *X, const int ldX, void *T
)
{
    if (val_type == VAL_TYPE_DOUBLE)
    {
        mpcg<double>(
            m, n, n_iter, A_mm, A, invM_mm, invM, 
            (const double *) B, ldB, (double *) X, ldX, (double *) T
        );
    }
    if (val_type == VAL_TYPE_FLOAT)
    {
        mpcg<float>(
            m, n, n_iter, A_mm, A, invM_mm, invM, 
            (const float *) B, ldB, (float *) X, ldX, (float *) T
        );
    }
}
