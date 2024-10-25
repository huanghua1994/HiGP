#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <limits>
#include <omp.h>

#include "nys_precond.h"
#include "../common.h"
#include "../cpu_linalg.hpp"
#include "../utils.h"
#include "../dense_kernel_matrix.h"

template<typename VT>
void nys_precond_build(
    const int val_type, const int krnl_id, const VT *param, const void *dnoise,
    const int npt, const int pt_dim, const VT *coord, const int ldc, 
    const int *perm, const int k, const int need_grad, nys_precond_p *np
)
{
    int n_grad = need_grad ? 3 : 0;
    size_t VT_bytes = sizeof(VT);

    nys_precond_p np_ = (nys_precond_p) malloc(sizeof(nys_precond_s));
    memset(np_, 0, sizeof(nys_precond_s));

    // Copy parameters
    int n = npt;
    VT eta = param[3];
    np_->val_type = val_type;
    np_->n        = npt;
    np_->k        = k;
    np_->perm     = (int *) malloc(sizeof(int) * npt);
    np_->eta      = (VT *)  malloc(sizeof(VT));
    memcpy(np_->perm, perm, sizeof(int) * npt);
    memcpy(np_->eta,  &eta, sizeof(VT));

    // 1. Compute K1 = [K11, K12]
    VT *pX = (VT*) malloc(sizeof(VT) * npt * pt_dim);
    // gather_matrix_cols works on row-major matrices, swap row & column parameters
    gather_matrix_cols(VT_bytes, pt_dim, npt, perm, (const void *) coord, ldc, (void *) pX, npt);
    dense_krnl_mat_p dk_K1 = NULL;
    dense_krnl_mat_init(
        k, npt, pX, npt, npt, pX, 
        (void *) param, dnoise, krnl_id, val_type, &dk_K1
    );
    VT *K1  = (VT *) malloc(sizeof(VT) * k * npt);
    VT *dK1 = (VT *) malloc(sizeof(VT) * k * npt * n_grad);
    ASSERT_PRINTF(K1 != NULL && dK1 != NULL, "Failed to allocate work array for %s\n", __FUNCTION__);
    VT *dK1dl = NULL, *dK1df = NULL, *dK1ds = NULL;
    if (n_grad == 3)
    {
        dK1dl = dK1;
        dK1df = dK1 + k * npt;
        dK1ds = dK1 + k * npt * 2;
    } else {
        dK1 = NULL;
    }
    // TODO: does K11 need to add dnoise?
    dense_krnl_mat_grad_eval(dk_K1, K1, dK1dl, dK1df, dK1ds);
    dense_krnl_mat_free(&dk_K1);
    // Copy dK1 to np->dK1, correct dK1ds (its diagonal should be 1)
    np_->K1  = K1;
    np_->dK1 = dK1;
    if (dK1ds != NULL)
        for (int i = 0; i < k; i++) dK1ds[i * k + i] = 1;
    free(pX);

    // 2. Compute U0 and L11
    // This K11 does not have the diagonal shift (that's what we need)
    VT *K11 = (VT *) malloc(sizeof(VT) * k * k);
    ASSERT_PRINTF(K11 != NULL, "Failed to allocate work array for %s\n", __FUNCTION__);
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < k * k; i++) K11[i] = K1[i];
    // Slightly shift the diagonal to make Nystrom stable
    // nu  = sqrt(n) * eps(norm(K1, 'fro'));
    VT nu = 0, K1_fnorm = 0;
    #pragma omp parallel for schedule(static) reduction(+:K1_fnorm)
    for (int i = 0; i < n * k; i++) K1_fnorm += K1[i] * K1[i];
    K1_fnorm = std::sqrt(K1_fnorm);
    nu = std::sqrt((VT) n) * K1_fnorm * std::numeric_limits<VT>::epsilon();
    // K11 = K11 + nu * eye(k);
    for (int i = 0; i < k; i++) K11[i * k + i] += nu;
    VT *L11  = (VT *) malloc(sizeof(VT) * k * k);
    VT *invL = (VT *) malloc(sizeof(VT) * k * k);
    VT *U0   = (VT *) malloc(sizeof(VT) * n * k);
    ASSERT_PRINTF(L11 != NULL && invL != NULL && U0 != NULL, "Failed to allocate work array for %s\n", __FUNCTION__);
    // L11 = chol(K11, 'lower');
    memcpy(L11, K11, sizeof(VT) * k * k);
    int info = 0;
    xpotrf_(lower, &k, L11, &k, &info);
    ASSERT_PRINTF(info == 0, "xPOTRF failed, info = %d\n", info);
    // U0 = K1' * inv(L11)';
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < k * k; i++) invL[i] = 0;
    for (int i = 0; i < k; i++) invL[i * k + i] = 1;
    VT v_zero = 0.0, v_one = 1.0;
    xtrsm_(left, lower, notrans, nonunit, &k, &k, &v_one, L11, &k, invL, &k);
    xgemm_(
        trans, trans, &n, &k, &k,
        &v_one, K1, &k, invL, &k,
        &v_zero, U0, &n
    );
    np_->L11 = L11;
    free(invL);

    // 3. Compute U matrix and S array
    VT *S = (VT *) malloc(sizeof(VT) * k);
    ASSERT_PRINTF(S != NULL, "Failed to allocate work array for %s\n", __FUNCTION__);
    //#define NYSTROM_SVD_DIRECT
    #ifdef NYSTROM_SVD_DIRECT
    // [U, S, ~] = svd(U0, 0);
    int lwork = -1;
    VT *work = NULL, lwork_query = 0;
    xgesvd_(
        overwrt, nocalc, &n, &k, U0, &n, S,
        NULL, &k, NULL, &k, &lwork_query, &lwork, &info
    );
    lwork = (int) lwork_query;
    work = (VT *) malloc(sizeof(VT) * lwork);
    xgesvd_(  \
        overwrt, nocalc, &n, &k, U0, &n, S,
        NULL, &k, NULL, &k, work, &lwork, &info
    );
    ASSERT_PRINTF(info == 0, "xGESVD failed, info = %d\n", info);
    free(work);
    np_->U = U0;
    #else
    // Use EVD is usually faster but may be less accurate
    // MKL with ICC 19.1.3 has a bug in LAPACK_GESVD so we have to use EVD instead
    // H = U0' * U0;
    // [V, S] = eig(H);
    VT *H = (VT *) malloc(sizeof(VT) * k * k);
    int lwork = -1;
    VT *work = NULL, lwork_query = 0;
    xsyrk_(upper, trans, &k, &n, &v_one, U0, &n, &v_zero, H, &k);
    xsyev_(vector, upper, &k, H, &k, S, &lwork_query, &lwork, &info);
    lwork = (int) lwork_query;
    work = (VT *) malloc(sizeof(VT) * lwork);
    xsyev_(vector, upper, &k, H, &k, S, work, &lwork, &info);
    ASSERT_PRINTF(info == 0, "xSYEV failed, info = %d\n", info);
    // S = sqrt(S);
    // V = V .* inv(S);
    VT *invS = (VT *) malloc(sizeof(VT) * k);
    VT max_S = S[k - 1];
    VT eps = std::numeric_limits<VT>::epsilon();
    VT S_min_tol = std::sqrt(max_S) * eps * 10.0;
    for (int i = 0; i < k; i++)
    {
        if (S[i] < 0) S[i] = max_S * eps * eps;  // Safeguard
        S[i] = std::sqrt(S[i]);
        invS[i] = 1.0 / S[i];
        if (S[i] < S_min_tol) invS[i] = 0;  // Truncate extremely small eigenvalues
    }
    #pragma omp parallel for schedule(static)
    for (int j = 0; j < k; j++)
    {
        VT *V_j = H + j * k;
        #pragma omp simd
        for (int i = 0; i < k; i++) V_j[i] *= invS[j];
    }
    // U = U0 * V;
    VT *U = (VT *) malloc(sizeof(VT) * n * k);
    ASSERT_PRINTF(U != NULL, "Failed to allocate work array for %s\n", __FUNCTION__);
    xgemm_(
        notrans, notrans, &n, &k, &k,
        &v_one, U0, &n, H, &k,
        &v_zero, U, &n
    );
    np_->U = U;
    free(H);
    free(work);
    free(invS);
    #endif

    // 4. Compute M array
    // S = max(S.^2 - nu, 0);
    // M = 1 ./ (S + diag_shift);
    VT *M = (VT *) malloc(sizeof(VT) * k);
    for (int i = 0; i < k; i++)
    {
        S[i] = S[i] * S[i] - nu;
        if (S[i] < 0) S[i] = 0;
        M[i] = 1.0 / (S[i] + eta);
    }
    np_->M = M;

    // 5. Compute logdet
    VT logdet = (VT) (n - k) * std::log(eta);
    for (int i = 0; i < k; i++) logdet += std::log(S[i] + eta);
    np_->logdet = (VT *) malloc(sizeof(VT));
    memcpy(np_->logdet, &logdet, sizeof(VT));

    if (need_grad) nys_precond_grad_trace(np_);

    *np = np_;
}

// Build a Nystrom approximation preconditioner for f^2 * (K(coord, coord, l) + s * I)
void nys_precond_build(
    const int val_type, const int krnl_id, const void *param, const void *dnoise,
    const int npt, const int pt_dim, const void *coord, const int ldc, 
    const int *perm, const int k, const int need_grad, nys_precond_p *np
)
{
    if (val_type == VAL_TYPE_DOUBLE)
    {
        nys_precond_build<double>(
            val_type, krnl_id, (const double *) param, dnoise,
            npt, pt_dim, (const double *) coord, ldc, 
            perm, k, need_grad, np
        );
    }
    if (val_type == VAL_TYPE_FLOAT)
    {
        nys_precond_build<float>(
            val_type, krnl_id, (const float *)  param, dnoise,
            npt, pt_dim, (const float *)  coord, ldc, 
            perm, k, need_grad, np
        );
    }
}

// Free an initialized nys_precond struct
void nys_precond_free(nys_precond_p *np)
{
    nys_precond_p np_ = *np;
    if (np_ == NULL) return;
    free(np_->U);
    free(np_->M);
    free(np_->eta);
    free(np_->L11);
    free(np_->K1);
    free(np_->dK1);
    free(np_->logdet);
    free(np_->gt);
    free(np_);
    *np = NULL;
}

template<typename VT>
void nys_precond_apply(nys_precond_p np, const int nvec, const VT *B, const int ldB, VT *C, const int ldC)
{
    int n     = np->n;
    int k     = np->k;
    int *perm = np->perm;
    VT *U     = (VT *) np->U;
    VT *M     = (VT *) np->M;

    int ldB_ = n, ldC_ = n, ldT = k;
    VT v_zero = 0.0, v_one = 1.0, v_neg_one = -1.0;
    VT *pB = (VT *) malloc(sizeof(VT) * n * nvec);
    VT *pC = (VT *) malloc(sizeof(VT) * n * nvec);
    VT *T  = (VT *) malloc(sizeof(VT) * k * nvec);

    // pB = B(perm, :);
    #pragma omp parallel for schedule(static)
    for (int j = 0; j < nvec; j++)
    {
        const VT *B_j = B + j * ldB;
        VT *pB_j = pB + j * ldB_;
        #pragma omp simd
        for (int i = 0; i < n; i++) pB_j[i] = B_j[perm[i]];
    }

    // T = U' * pB;
    xgemm_(
        trans, notrans, &k, &nvec, &n,
        &v_one, U, &n, pB, &ldB_,
        &v_zero, T, &ldT  \
    );

    // pC = pB - U * T;
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n * nvec; i++) pC[i] = pB[i];
    xgemm_(
        notrans, notrans, &n, &nvec, &k,
        &v_neg_one, U, &n, T, &ldT,
        &v_one, pC, &ldC_
    );

    // T = diag(M) * T, M .* each column of T
    for (int j = 0; j < nvec; j++)
    {
        VT *T_j = T + j * ldT;
        #pragma omp simd
        for (int i = 0; i < k; i++) T_j[i] *= M[i];
    }

    // pC = pC ./ eta + U * T;
    VT eta = *((VT *) np->eta);
    VT inv_eta = 1.0 / eta;
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n * nvec; i++) pC[i] *= inv_eta;
    xgemm_(
        notrans, notrans, &n, &nvec, &k,
        &v_one, U, &n, T, &ldT,
        &v_one, pC, &ldC_
    );

    // C(perm, :) = pC;
    #pragma omp parallel for schedule(static)
    for (int j = 0; j < nvec; j++)
    {
        VT *pC_j = pC + j * ldC_;
        VT *C_j  = C  + j * ldC;
        for (int i = 0; i < n; i++) C_j[perm[i]] = pC_j[i];
    }

    free(pB);
    free(pC);
    free(T);
}

// Apply the Nystrom preconditioner to multiple column vectors
void nys_precond_apply(const void *np, const int n, const void *B, const int ldB, void *C, const int ldC)
{
    nys_precond_p np_ = (nys_precond_p) np;
    if (np_ == NULL) return;
    if (np_->val_type == VAL_TYPE_DOUBLE) nys_precond_apply<double>(np_, n, (const double *) B, ldB, (double *) C, ldC);
    if (np_->val_type == VAL_TYPE_FLOAT)  nys_precond_apply<float> (np_, n, (const float *)  B, ldB, (float *)  C, ldC);
}

template<typename VT>
void nys_precond_dapply(nys_precond_p np, const int nvec, const VT *X, const int ldX, VT *Y, const int ldY, const int skip_perm)
{
    int n_grad = 3;
    int n     = np->n;
    int k     = np->k;
    int *perm = np->perm;
    VT  *K1   = (VT *) np->K1;
    VT  *dK1  = (VT *) np->dK1;
    VT  *L11  = (VT *) np->L11;

    VT *pX = (VT *) malloc(sizeof(VT) * n * nvec);
    VT *pY = (VT *) malloc(sizeof(VT) * n * nvec);
    ASSERT_PRINTF(pX != NULL && pY != NULL, "Failed to allocate work array for %s\n", __FUNCTION__);

    if (skip_perm)
    {
        #pragma omp parallel for schedule(static)
        for (int j = 0; j < nvec; j++)
        {
            const VT *X_j = X + j * ldX;
            VT *pX_j = pX + j * n;
            #pragma omp simd
            for (int i = 0; i < n; i++) pX_j[i] = X_j[i];
        }
    } else {
        #pragma omp parallel for schedule(static)
        for (int j = 0; j < nvec; j++)
        {
            const VT *X_j = X + j * ldX;
            VT *pX_j = pX + j * n;
            #pragma omp simd
            for (int i = 0; i < n; i++) pX_j[i] = X_j[perm[i]];
        }
    }  // End of "if (skip_perm)"

    // K1X = K1 * pX;
    // iK11_K1X = L11' \ (L11 \ K1X);
    VT *iK11_K1X = (VT *) malloc(sizeof(VT) * k * nvec);
    ASSERT_PRINTF(iK11_K1X != NULL, "Failed to allocate work array for %s\n", __FUNCTION__);
    VT v_zero = 0.0, v_one = 1.0;
    xgemm_(
        notrans, notrans, &k, &nvec, &n,
        &v_one, K1, &k, pX, &n,
        &v_zero, iK11_K1X, &k
    );
    xtrsm_(left, lower, notrans, nonunit, &k, &nvec, &v_one, L11, &k, iK11_K1X, &k);
    xtrsm_(left, lower, trans,   nonunit, &k, &nvec, &v_one, L11, &k, iK11_K1X, &k);

    VT *T3 = (VT *) malloc(sizeof(VT) * k * nvec * 2);
    VT *T5 = (VT *) malloc(sizeof(VT) * n * nvec * 2);
    ASSERT_PRINTF(T3 != NULL && T5 != NULL, "Failed to allocate work array for %s\n", __FUNCTION__);
    VT *T1 = T3, *T2 = T3 + k * nvec, *T4 = T3;
    VT *T5_l = T5, *T5_r = T5 + n * nvec;
    int nvec2 = nvec * 2;
    for (int i_grad = 0; i_grad < n_grad; i_grad++)
    {
        VT *Y_i = Y + i_grad * ldY * nvec;
        VT *dK1_i  = dK1 + i_grad * k * n;
        VT *dK11_i = dK1_i;
        // pY = dK1{i}' * iK11_K1X;
        xgemm_(
            trans, notrans, &n, &nvec, &k,
            &v_one, dK1_i, &k, iK11_K1X, &k,
            &v_zero, pY, &n
        );
        // T1 = dK11_i * iK11_K1X;
        xgemm_(
            notrans, notrans, &k, &nvec, &k,
            &v_one, dK11_i, &k, iK11_K1X, &k,
            &v_zero, T1, &k
        );
        // T2 = dK1{i} * pX;
        xgemm_(
            notrans, notrans, &k, &nvec, &n,
            &v_one, dK1_i, &k, pX, &n,
            &v_zero, T2, &k
        );
        // T3 = [T1, T2];
        // T4 = L11' \ (L11 \ T3);
        xtrsm_(left, lower, notrans, nonunit, &k, &nvec2, &v_one, L11, &k, T3, &k);
        xtrsm_(left, lower, trans,   nonunit, &k, &nvec2, &v_one, L11, &k, T3, &k);
        // T5 = K1' * T4;
        xgemm_(
            trans, notrans, &n, &nvec2, &k,
            &v_one, K1, &k, T4, &k,
            &v_zero, T5, &n
        );
        // pY = pY - T5(:, 1 : nvec) + T5(:, nvec + 1 : 2 * nvec);
        #pragma omp parallel for schedule(static)
        for (int j = 0; j < nvec; j++)
        {
            VT *pY_j   = pY   + j * n;
            VT *T5_l_j = T5_l + j * n;
            VT *T5_r_j = T5_r + j * n;
            #pragma omp simd
            for (int i = 0; i < n; i++) pY_j[i] = pY_j[i] - T5_l_j[i] + T5_r_j[i];
        }
        if (skip_perm)
        {
            #pragma omp parallel for schedule(static)
            for (int j = 0; j < nvec; j++)
            {
                VT *pY_j = pY + j * n;
                VT *Y_i_j = Y_i + j * ldY;
                #pragma omp simd
                for (int i = 0; i < n; i++) Y_i_j[i] = pY_j[i];
            }
        } else {
            #pragma omp parallel for schedule(static)
            for (int j = 0; j < nvec; j++)
            {
                VT *pY_j = pY + j * n;
                VT *Y_i_j = Y_i + j * ldY;
                #pragma omp simd
                for (int i = 0; i < n; i++) Y_i_j[perm[i]] = pY_j[i];
            }
        }  // End of "if (skip_perm)"
    }  // End of i_grad loop

    free(pX);
    free(pY);
    free(iK11_K1X);
    free(T3);
    free(T5);
}

// Compute Y := d M^{-1} / d {l, f, s} * X, M is the Nystrom precond matrix
void nys_precond_dapply(nys_precond_p np, const int n, const void *X, const int ldX, void *Y, const int ldY, const int skip_perm)
{
    if (np == NULL) return;
    if (np->val_type == VAL_TYPE_DOUBLE) nys_precond_dapply<double>(np, n, (const double *) X, ldX, (double *) Y, ldY, skip_perm);
    if (np->val_type == VAL_TYPE_FLOAT)  nys_precond_dapply<float> (np, n, (const float *)  X, ldX, (float *)  Y, ldY, skip_perm);
}

template<typename VT>
void nys_precond_grad_trace(nys_precond_p np, VT *gt)
{
    int n_grad = 3;
    int n   = np->n;
    int k   = np->k;
    VT *K1  = (VT *) np->K1;
    VT *L11 = (VT *) np->L11;
    VT *dK1 = (VT *) np->dK1;
    VT eta  = *((VT *) np->eta);

    // L = K1' * inv(L11');
    VT *invL = (VT *) malloc(sizeof(VT) * k * k);
    VT *L = (VT *) malloc(sizeof(VT) * n * k);
    ASSERT_PRINTF(invL != NULL && L != NULL, "Failed to allocate work array for %s\n", __FUNCTION__);
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < k * k; i++) invL[i] = 0;
    for (int i = 0; i < k; i++) invL[i * k + i] = 1;
    VT v_zero = 0.0, v_one = 1.0;
    xtrsm_(left, lower, notrans, nonunit, &k, &k, &v_one, L11, &k, invL, &k);
    xgemm_(
        trans, trans, &n, &k, &k,
        &v_one, K1, &k, invL, &k,
        &v_zero, L, &n
    );

    VT *dL  = (VT *) malloc(sizeof(VT) * n * k);
    VT *T   = (VT *) malloc(sizeof(VT) * k * k);
    VT *dKL = (VT *) malloc(sizeof(VT) * n * k);
    ASSERT_PRINTF(dL != NULL && T != NULL && dKL != NULL, "Failed to allocate work array for %s\n", __FUNCTION__);
    for (int i_grad = 0; i_grad < n_grad; i_grad++)
    {
        VT *dK1_i  = dK1 + i_grad * k * n;
        VT *dK11_i = dK1_i;
        // dL = dK1{i}' * inv(L11)'
        xgemm_(
            trans, trans, &n, &k, &k,
            &v_one, dK1_i, &k, invL, &k,
            &v_zero, dL, &n
        );
        // T = dK11_i * inv(L11)';
        xgemm_(
            notrans, trans, &k, &k, &k,
            &v_one, dK11_i, &k, invL, &k,
            &v_zero, T, &k
        );
        // T = L11 \ T;
        xtrsm_(left, lower, notrans, nonunit, &k, &k, &v_one, L11, &k, T, &k);
        // dKL = L * T;
        xgemm_(
            notrans, notrans, &n, &k, &k,
            &v_one, L, &n, T, &k,
            &v_zero, dKL, &n
        );
        // gt(i) = 2 * sum(sum(dL .* L)) - sum(sum(dKL .* L));
        VT val0 = 0, val1 = 0;
        #pragma omp parallel for schedule(static) reduction(+: val0, val1)
        for (int i = 0; i < n * k; i++)
        {
            val0 += dL[i] * L[i];
            val1 += dKL[i] * L[i];
        }
        gt[i_grad] = 2.0 * val0 - val1;
    }  // End of i_grad loop

    // T1 = eta * eye(k) + L' * L;
    VT *T1 = T, *LP = dL, *invT1 = dKL;  // T, dL, and dKL are not used anymore
    VT *dLP = (VT *) malloc(sizeof(VT) * n * k * n_grad);
    ASSERT_PRINTF(dLP != NULL, "Failed to allocate work array for %s\n", __FUNCTION__);
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < k * k; i++)
    {
        T1[i] = 0;
        invT1[i] = 0;
    }
    for (int i = 0; i < k; i++)
    {
        T1[i * k + i] = eta;
        invT1[i * k + i] = 1.0;
    }
    xgemm_(
        trans, notrans, &k, &k, &n,
        &v_one, L, &n, L, &n,
        &v_one, T1, &k
    );
    // LP = L * inv(T1);
    int info;
    xposv_(lower, &k, &k, T1, &k, invT1, &k, &info);
    ASSERT_PRINTF(info == 0, "xPOSV failed, info = %d\n", info);
    xgemm_(
        notrans, notrans, &n, &k, &k,
        &v_one, L, &n, invT1, &k,
        &v_zero, LP, &n
    );

    nys_precond_dapply(np, k, L, n, dLP, n, 1);
    for (int i_grad = 0; i_grad < n_grad; i_grad++)
    {
        // gt(i) = (gt(i) - sum(sum(dLP{i} .* LP))) / eta;
        VT val = 0;
        VT *dLP_i = dLP + i_grad * n * k;
        #pragma omp parallel for schedule(static) reduction(+: val)
        for (int i = 0; i < n * k; i++) val += dLP_i[i] * LP[i];
        gt[i_grad] = (gt[i_grad] - val) / eta;
    }

    free(invL);
    free(L);
    free(dL);
    free(T);
    free(dKL);
    free(dLP);
}

// Compute the trace of M^{-1} * dM / d {l, f, s}, store the result in np->gt
void nys_precond_grad_trace(nys_precond_p np)
{
    if (np == NULL) return;
    if (np->gt != NULL) return;
    size_t VT_bytes = (np->val_type == VAL_TYPE_DOUBLE) ? sizeof(double) : sizeof(float);
    np->gt = (void *) malloc(VT_bytes * 3);
    if (np->val_type == VAL_TYPE_DOUBLE) nys_precond_grad_trace<double>(np, (double *) np->gt);
    if (np->val_type == VAL_TYPE_FLOAT)  nys_precond_grad_trace<float> (np, (float *)  np->gt);
}