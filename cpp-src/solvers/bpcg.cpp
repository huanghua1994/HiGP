#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <omp.h>

#include "bpcg.h"
#include "../utils.h"
#include "../cpu_linalg.hpp"

template<typename VT>
static void bpcg(
    const int m, const int n, const int max_iter, const VT *reltol_,
    matmul_fptr A_mm, const void *A, matmul_fptr invM_mm, const void *invM, 
    const VT *B, const int ldB, VT *X, const int ldX, int *iters_
)
{
    const size_t val_bytes = sizeof(VT);
    const VT reltol = *reltol_;
    const int ldR = m, ldZ = m, ldP = m, ldS = m, use_omp = 1;
    VT *workmat = (VT *) malloc(sizeof(VT) * 8 * m * n);
    VT *R     = workmat + m * n * 0;  // Residual vectors
    VT *Z     = workmat + m * n * 1;  // Preconditioned residual vectors
    VT *P     = workmat + m * n * 2;  // Search direction vectors
    VT *S     = workmat + m * n * 3;  // A * P
    VT *R_tmp = workmat + m * n * 4;
    VT *Z_tmp = workmat + m * n * 5;
    VT *P_tmp = workmat + m * n * 6;
    VT *S_tmp = workmat + m * n * 7;
    VT *vr0   = (VT *) malloc(sizeof(VT) * n);
    VT *cnorm = (VT *) malloc(sizeof(VT) * n);
    VT *snorm = (VT *) malloc(sizeof(VT) * n);
    int *rhs_active = (int *) malloc(sizeof(int) * n);
    int *active_idx = (int *) malloc(sizeof(int) * n);
    int *iters      = (int *) malloc(sizeof(int) * n);

    // Unnecessary initialization, just to silence GCC's Wmaybe-uninitialized
    #pragma omp parallel for if(use_omp == 1) schedule(static)
    for (int i = 0; i < m * n; i++) R[i] = 0.0;

    // R = B - A_mm(X);
    // S = zeros(m, nrhs);
    A_mm(A, n, X, ldX, R, ldR);
    #pragma omp parallel for schedule(static)
    for (int j = 0; j < n; j++)
    {
        VT *Rj = R + j * ldR;
        VT *Sj = S + j * ldS;
        const VT *Bj = B + j * ldB;
        #pragma omp simd
        for (int i = 0; i < m; i++) Rj[i] = Bj[i] - Rj[i];
        memset(Sj, 0, sizeof(VT) * m);
    }
    // Z = invM_mm(R);
    if (invM_mm != NULL) invM_mm(invM, n, R, ldR, Z, ldZ);
    else copy_matrix(val_bytes, n, m, R, ldR, Z, ldZ, use_omp);  // copy_matrix uses row-major
    // P = Z;
    copy_matrix(val_bytes, n, m, Z, ldZ, P, ldP, use_omp);

    // Compute the stopping norm of each RHS, set all RHS as active
    int n_active = n, i_one = 1;
    for (int j = 0; j < n; j++)
    {
        rhs_active[j] = 1;
        active_idx[j] = j;
        iters[j]      = -1;
        const VT *Bj = B + j * ldB;
        snorm[j] = reltol * xnrm2_(&m, Bj, &i_one);
    }

    // Main loop
    for (int i = 0; i < max_iter; i++)
    {
        // P_tmp = P(:, active_idx);
        // S_tmp = Afun_mm(P_tmp);
        // S(:, active_idx) = S_tmp;
        #pragma omp parallel for schedule(static)
        for (int ja = 0; ja < n_active; ja++)
            memcpy(P_tmp + ja * ldP, P + active_idx[ja] * ldP, sizeof(VT) * m);
        A_mm(A, n_active, P_tmp, ldP, S_tmp, ldS);
        #pragma omp parallel for schedule(static)
        for (int ja = 0; ja < n_active; ja++)
            memcpy(S + active_idx[ja] * ldS, S_tmp + ja * ldS, sizeof(VT) * m);

        #pragma omp parallel for schedule(static)
        for (int ja = 0; ja < n_active; ja++)
        {
            int j = active_idx[ja];
            // vr0(j) = dot(R(:, j), Z(:, j));
            // alpha  = vr0(j) / dot(S(:, j), P(:, j));
            VT alpha = 0.0;
            VT *Rj = R + j * ldR;
            VT *Zj = Z + j * ldZ;
            VT *Sj = S + j * ldS;
            VT *Pj = P + j * ldP;
            vr0[j] = 0.0;
            #pragma omp simd
            for (int l = 0; l < m; l++)
            {
                vr0[j] += Rj[l] * Zj[l];
                alpha  += Sj[l] * Pj[l];
            }
            alpha = vr0[j] / alpha;

            // X(:, j) = X(:, j) + a .* P(:, j);
            // R(:, j) = R(:, j) - a .* S(:, j);
            VT *Xj = X + j * ldX;
            #pragma omp simd 
            for (int l = 0; l < m; l++)
            {
                Xj[l] += alpha * Pj[l];
                Rj[l] -= alpha * Sj[l];
            }
            // cnorm(j) = norm(R(:, j));
            cnorm[j] = xnrm2_(&m, Rj, &i_one);
        }  // End of ja loop

        // Count the numer of active RHS after update
        n_active = 0;
        for (int j = 0; j < n; j++)
        {
            if (cnorm[j] >= snorm[j])
            {
                active_idx[n_active] = j;
                n_active++;
            } else {
                if (iters[j] == -1) iters[j] = i;
            }
        }
        if (n_active == 0) break;

        if (i == max_iter - 1) break;
        // R_tmp = R(:, active_idx);
        #pragma omp parallel for schedule(static)
        for (int ja = 0; ja < n_active; ja++)
            memcpy(R_tmp + ja * ldR, R + active_idx[ja] * ldR, sizeof(VT) * m);
        // Z_tmp = invM_mm(R_tmp);
        if (invM_mm != NULL) invM_mm(invM, n_active, R_tmp, ldR, Z_tmp, ldZ);
        else copy_matrix(val_bytes, n_active, m, R_tmp, ldR, Z_tmp, ldZ, use_omp);
        // Z(:, active_idx) = Z_tmp;
        #pragma omp parallel for schedule(static)
        for (int ja = 0; ja < n_active; ja++)
            memcpy(Z + active_idx[ja] * ldZ, Z_tmp + ja * ldZ, sizeof(VT) * m);

        #pragma omp parallel for schedule(static)
        for (int ja = 0; ja < n_active; ja++)
        {
            int j = active_idx[ja];
            // beta = dot(R(:, j), Z(:, j)) / vr0(j);
            VT beta = 0.0;
            VT *Rj = R + j * ldR;
            VT *Zj = Z + j * ldZ;
            #pragma omp simd
            for (int l = 0; l < m; l++) beta += Rj[l] * Zj[l];
            beta /= vr0[j];
            // P(:, j) = Z(:, j) + beta .* P(:, j);
            VT *Pj = P + j * ldP;
            #pragma omp simd
            for (int l = 0; l < m; l++) Pj[l] = Zj[l] + beta * Pj[l];
        }
    }  // End of i loop

    if (iters_ != NULL) memcpy(iters_, iters, sizeof(int) * n);
    free(workmat);
    free(vr0);
    free(cnorm);
    free(snorm);
    free(rhs_active);
    free(active_idx);
    free(iters);
}

// Blocked (multiple RHS) preconditioned conjugate gradient
void bpcg(
    const int m, const int n, const int val_type, const int max_iter, const void *reltol, 
    matmul_fptr A_mm, const void *A, matmul_fptr invM_mm, const void *invM, 
    const void *B, const int ldB, void *X, const int ldX, int *iters
)
{
    if (val_type == VAL_TYPE_DOUBLE)
    {
        bpcg<double>(
            m, n, max_iter, (const double *) reltol, 
            A_mm, A, invM_mm, invM, 
            (const double *) B, ldB, (double *) X, ldX, iters
        );
    }
    if (val_type == VAL_TYPE_FLOAT)
    {
        bpcg<float>(
            m, n, max_iter, (const float *) reltol, 
            A_mm, A, invM_mm, invM, 
            (const float *) B, ldB, (float *) X, ldX, iters
        );
    }
}
