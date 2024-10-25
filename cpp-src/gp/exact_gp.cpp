#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <omp.h>
#include <time.h>

#include "../common.h"
#include "../dense_kernel_matrix.h"
#include "../cpu_linalg.hpp"
#include "../utils.h"
#include "exact_gp.h"
#include "nonneg_transform.h"
#include "gpc_common.h"

template<typename VT>
static void exact_gpr_loss_compute(
    const int val_type, const int nnt_id, const int krnl_id, const VT *param, 
    const int n_train, const int pt_dim, const VT *X_train, const int ldX, 
    const VT *Y_train, const void *dnoise, VT *L, VT *L_grad
)
{
    size_t VT_bytes = sizeof(VT);

    // 1. Transform the hyper-parameters
    VT nnt_param[4], nnt_dparam[4];
    nnt_param[0] = param[0];
    for (int i = 1; i < 4; i++) 
        nonneg_transform(val_type, nnt_id, &param[i], &nnt_param[i], &nnt_dparam[i]);

    // 2. Compute the dense kernel matrix and its eigen decomposition
    dense_krnl_mat_p dkmat = NULL;
    dense_krnl_mat_init(
        n_train, ldX, (const void *) X_train, 
        n_train, ldX, (const void *) X_train, 
        nnt_param, dnoise, krnl_id, val_type, &dkmat
    );
    VT *K = (VT *) malloc(VT_bytes * n_train * n_train);
    VT *w = (VT *) malloc(VT_bytes * n_train);
    ASSERT_PRINTF(K != NULL && w != NULL, "Failed to allocate work arrays for %s\n", __FUNCTION__);
    dense_krnl_mat_grad_eval(dkmat, (void *) K, NULL, NULL, NULL);
    int lwork = -1, liwork = -1, info = 0, iwork_query = 0, *iwork = NULL;
    VT work_query = 0, *work = NULL;
    xsyevd_(
        vector, lower, &n_train, K, &n_train, 
        w, &work_query, &lwork, &iwork_query, &liwork, &info
    );
    lwork = ((int) work_query) + 100;  // +100 to prevent VT to int rounding makes it insufficient
    liwork = iwork_query;
    work = (VT *) malloc(VT_bytes * lwork);
    iwork = (int *) malloc(sizeof(int) * liwork);
    ASSERT_PRINTF(
        work != NULL && iwork != NULL,
        "Failed to allocate work arrays for %s\n", __FUNCTION__
    );
    xsyevd_(
        vector, lower, &n_train, K, &n_train, 
        w, work, &lwork, iwork, &liwork, &info
    );
    ASSERT_PRINTF(info == 0, "xSYEVD returned error code %d\n", info);

    // 3. Compute K^{-1} using the eigen decomposition
    VT v_one = 1.0, v_zero = 0.0;
    VT *V_invD = (VT *) malloc(VT_bytes * n_train * n_train);
    VT *invK   = (VT *) malloc(VT_bytes * n_train * n_train);
    ASSERT_PRINTF(V_invD != NULL && invK != NULL, "Failed to allocate work arrays for %s\n", __FUNCTION__);
    #pragma omp parallel for schedule(static)
    for (int j = 0; j < n_train; j++)
    {
        VT *V_j      = K     + j * n_train;
        VT *V_invD_j = V_invD + j * n_train;
        VT invD_j    = 1.0 / w[j];
        #pragma omp simd
        for (int i = 0; i < n_train; i++) V_invD_j[i] = V_j[i] * invD_j;
    }
    xgemm_(
        notrans, trans, &n_train, &n_train, &n_train,
        &v_one, V_invD, &n_train, K, &n_train, 
        &v_zero, invK, &n_train
    );

    // 4. Compute the GP loss
    int i_one = 1;
    VT *iKY = (VT *) malloc(VT_bytes * n_train);
    ASSERT_PRINTF(iKY != NULL, "Failed to allocate work arrays for %s\n", __FUNCTION__);
    xgemv_(
        notrans, &n_train, &n_train, &v_one, invK, &n_train,
        Y_train, &i_one, &v_zero, iKY, &i_one
    );
    // L1 = Y' * inv(K) * Y;
    // L2 = sum(log(abs(eig(D))));
    VT L1 = 0, L2 = 0;
    #pragma omp simd
    for (int i = 0; i < n_train; i++)
    {
        L1 += Y_train[i] * iKY[i];
        L2 += std::log(std::abs(w[i]));
    }
    *L = 0.5 * (L1 + L2 + (VT) n_train * std::log(2.0 * M_PI));

    // 5. Compute the GP loss derivatives w.r.t. l, f, and s
    VT *dK = (VT *) malloc(VT_bytes * n_train * n_train);
    VT *dv = (VT *) malloc(VT_bytes * n_train);
    ASSERT_PRINTF(dK != NULL && dv != NULL, "Failed to allocate work arrays for %s\n", __FUNCTION__);
    void *dKdl = NULL, *dKds = NULL, *dKdf = NULL;
    for (int d = 0; d < 3; d++)
    {
        VT L1_grad = 0, L2_grad = 0;
        if (d == 0) { dKdl = (void *) dK; dKdf = NULL; dKds = NULL; }
        if (d == 1) { dKdl = NULL; dKdf = (void *) dK; dKds = NULL; }
        if (d == 2) { dKdl = NULL; dKdf = NULL; dKds = (void *) dK; }
        dense_krnl_mat_grad_eval(dkmat, NULL, dKdl, dKdf, dKds);

        // L1_grad(d) = iKY' * dK * iKY * dparam(d);
        xgemv_(
            notrans, &n_train, &n_train, &v_one, dK, &n_train, 
            iKY, &i_one, &v_zero, dv, &i_one
        );
        #pragma omp simd
        for (int i = 0; i < n_train; i++) L1_grad += iKY[i] * dv[i];
        L1_grad *= nnt_dparam[d + 1];

        // L2_grad(d) = sum(sum(invK' .* dK)) * dparam(d);
        #pragma omp parallel for schedule(static) reduction(+:L2_grad)
        for (int j = 0; j < n_train; j++)
        {
            VT *invK_j = invK + j * n_train;
            VT *dK_j   = dK   + j * n_train;
            VT res = 0;
            #pragma omp simd
            for (int i = 0; i < n_train; i++) res += invK_j[i] * dK_j[i];
            L2_grad += res;
        }
        L2_grad *= nnt_dparam[d + 1];

        L_grad[d] = 0.5 * (L2_grad - L1_grad);
    }  // End of d loop

    dense_krnl_mat_free(&dkmat);
    free(K);
    free(w);
    free(V_invD);
    free(invK);
    free(iKY);
    free(dK);
    free(dv);
    free(work);
    free(iwork);
}

// Compute the GP loss and its derivatives w.r.t. l, f, and s using exact solve
void exact_gpr_loss_compute(
    const int val_type, const int nnt_id, const int krnl_id, const void *param, 
    const int n_train, const int pt_dim, const void *X_train, const int ldX, 
    const void *Y_train, void *L, void *L_grad
)
{
    if (val_type == VAL_TYPE_DOUBLE)
    {
        exact_gpr_loss_compute<double>(
            val_type, nnt_id, krnl_id, (const double *) param,
            n_train, pt_dim, (const double *) X_train, ldX,
            (const double *) Y_train, NULL, (double *) L, (double *) L_grad
        );
    }
    if (val_type == VAL_TYPE_FLOAT)
    {
        exact_gpr_loss_compute<float>(
            val_type, nnt_id, krnl_id, (const float *)  param,
            n_train, pt_dim, (const float *)  X_train, ldX,
            (const float *)  Y_train, NULL, (float *)  L, (float *)  L_grad
        );
    }
}

template<typename VT>
static void exact_gpc_loss_compute(
    const int val_type, const int nnt_id, const int krnl_id, const VT *params, 
    const int n_train, const int pt_dim, const VT *X_train, const int ldX, 
    const int *Y_train, const int n_class, VT *L, VT *L_grads
)
{
    VT *dnoises = NULL, *Ys = NULL;
    gpc_process_label(val_type, n_train, n_class, Y_train, (void **) &dnoises, (void **) &Ys);

    VT inv_n = 1.0 / (VT) n_train;
    VT Li = 0, Li_grad[3] = {0, 0, 0};
    VT param[4] = {(VT) pt_dim, 0, 0, 0};
    *L = 0.0;
    for (int i_class = 0; i_class < n_class; i_class++)
    {
        VT *dnoise_i = dnoises + i_class * n_train;
        VT *Y_i      = Ys      + i_class * n_train;
        param[1] = params[i_class];
        param[2] = params[i_class + n_class];
        param[3] = params[i_class + n_class * 2];
        exact_gpr_loss_compute<VT>(
            val_type, nnt_id, krnl_id, &param[0],
            n_train, pt_dim, X_train, ldX,
            Y_i, (const void *) dnoise_i, &Li, Li_grad
        );
        *L += Li * inv_n;
        L_grads[i_class]               = Li_grad[0] * inv_n;
        L_grads[i_class + n_class]     = Li_grad[1] * inv_n;
        L_grads[i_class + n_class * 2] = Li_grad[2] * inv_n;
    }

    free(Ys);
    free(dnoises);
}

// Compute GP classification loss and its derivatives w.r.t. [l, f, s] using exact solve
void exact_gpc_loss_compute(
    const int val_type, const int nnt_id, const int krnl_id, const void *params, 
    const int n_train, const int pt_dim, const void *X_train, const int ldX, 
    const int *Y_train, const int n_class, void *L, void *L_grads
)
{
    if (val_type == VAL_TYPE_DOUBLE)
    {
        exact_gpc_loss_compute<double>(
            val_type, nnt_id, krnl_id, (const double *) params,
            n_train, pt_dim, (const double *) X_train, ldX,
            Y_train, n_class, (double *) L, (double *) L_grads
        );
    }
    if (val_type == VAL_TYPE_FLOAT)
    {
        exact_gpc_loss_compute<float>(
            val_type, nnt_id, krnl_id, (const float *)  params,
            n_train, pt_dim, (const float *)  X_train, ldX,
            Y_train, n_class, (float *)  L, (float *)  L_grads
        );
    }
}

template<typename VT>
static void exact_gpr_predict(
    const int val_type, const int nnt_id, const int krnl_id, const VT *param, 
    const int n_train, const int pt_dim, const VT *X_train, const int ldX, 
    const VT *Y_train, const int n_pred, const VT *X_pred, const int ldXp, 
    VT *Y_pred, VT *stddev, VT *cov2, VT *dnoise
)
{
    size_t VT_bytes = sizeof(VT);

    // 1. Transform the hyper-parameters
    VT nnt_param[4], nnt_dparam[4];
    nnt_param[0] = param[0];
    for (int i = 1; i < 4; i++) 
        nonneg_transform(val_type, nnt_id, &param[i], &nnt_param[i], &nnt_dparam[i]);

    // 2. Compute kernel matrix and its derivatives
    dense_krnl_mat_p dk11 = NULL, dk12 = NULL, dk22 = NULL;
    // K11 = kernel(X_train, X_train, dnoise);
    dense_krnl_mat_init(
        n_train, ldX, (const void *) X_train, 
        n_train, ldX, (const void *) X_train, 
        nnt_param, dnoise, krnl_id, val_type, &dk11
    );
    // K12 = kernel(X_train, X_pred,  []);
    dense_krnl_mat_init(
        n_train, ldX, (const void *) X_train, 
        n_pred, ldXp, (const void *) X_pred, 
        nnt_param, NULL, krnl_id, val_type, &dk12
    );
    // K22 = kernel(X_pred,  X_pred,  zeros(n_pred, 1));
    dense_krnl_mat_init(
        n_pred, ldXp, (const void *) X_pred, 
        n_pred, ldXp, (const void *) X_pred, 
        nnt_param, NULL, krnl_id, val_type, &dk22
    );
    VT *K11 = (VT *) malloc(VT_bytes * n_train * n_train);
    VT *K12 = (VT *) malloc(VT_bytes * n_train * n_pred);
    VT *K22 = (VT *) malloc(VT_bytes * n_pred  * n_pred);
    ASSERT_PRINTF(K11 != NULL && K12 != NULL && K22 != NULL, "Failed to allocate work arrays for %s\n", __FUNCTION__);
    dense_krnl_mat_grad_eval(dk11, (void *) K11, NULL, NULL, NULL);
    dense_krnl_mat_grad_eval(dk12, (void *) K12, NULL, NULL, NULL);
    dense_krnl_mat_grad_eval(dk22, (void *) K22, NULL, NULL, NULL);

    // 3. iK11 = inv(K11);
    //    iK11_Y = iK11 * Y_train;
    //    Y_pred = K12' * iK11_Y;
    int info = 0, i_one = 1;
    VT v_one = 1.0, v_zero = 0.0;
    VT *iKY = (VT *) malloc(VT_bytes * n_train);
    ASSERT_PRINTF(iKY != NULL, "Failed to allocate work arrays for %s\n", __FUNCTION__);
    memcpy(iKY, Y_train, VT_bytes * n_train);
    xpotrf_(lower, &n_train, K11, &n_train, &info);
    ASSERT_PRINTF(info == 0, "xPOTRF returned error code %d\n", info);
    xpotrs_(lower, &n_train, &i_one, K11, &n_train, iKY, &n_train, &info);
    ASSERT_PRINTF(info == 0, "xPOTRS returned error code %d\n", info);
    xgemv_(
        trans, &n_train, &n_pred, &v_one, K12, &n_train,
        iKY, &i_one, &v_zero, Y_pred, &i_one
    );

    // 4. iK11_K12 = iK11 * K12;
    //    cov2 = K22 - K12' * iK11_K12;
    //    stddev = sqrt(abs(diag(cov2)));
    VT v_neg_one = -1.0;
    VT *iK11_K12 = (VT *) malloc(VT_bytes * n_train * n_pred);
    ASSERT_PRINTF(iK11_K12 != NULL, "Failed to allocate work arrays for %s\n", __FUNCTION__);
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n_train * n_pred; i++) iK11_K12[i] = K12[i];
    xpotrs_(lower, &n_train, &n_pred, K11, &n_train, iK11_K12, &n_train, &info);
    ASSERT_PRINTF(info == 0, "xPOTRS returned error code %d\n", info);
    xgemm_(
        trans, notrans, &n_pred, &n_pred, &n_train,
        &v_neg_one, K12, &n_train, iK11_K12, &n_train,
        &v_one, K22, &n_pred
    );
    for (int i = 0; i < n_pred; i++) stddev[i] = std::sqrt(std::abs(K22[i * n_pred + i]));

    if (cov2 != NULL) memcpy(cov2, K22, sizeof(VT) * n_pred * n_pred);

    dense_krnl_mat_free(&dk11);
    dense_krnl_mat_free(&dk12);
    dense_krnl_mat_free(&dk22);
    free(K11);
    free(K12);
    free(K22);
    free(iKY);
    free(iK11_K12);
}

// GP prediction with a given kernel and its parameters using exact solve
void exact_gpr_predict(
    const int val_type, const int nnt_id, const int krnl_id, const void *param, 
    const int n_train, const int pt_dim, const void *X_train, const int ldX, 
    const void *Y_train, const int n_pred, const void *X_pred, const int ldXp, 
    void *Y_pred, void *stddev
)
{
    if (val_type == VAL_TYPE_DOUBLE)
    {
        exact_gpr_predict<double>(
            val_type, nnt_id, krnl_id, (const double *) param,
            n_train, pt_dim, (const double *) X_train, ldX,
            (const double *) Y_train, n_pred, (const double *) X_pred, ldXp,
            (double *) Y_pred, (double *) stddev, NULL, NULL
        );
    }
    if (val_type == VAL_TYPE_FLOAT)
    {
        exact_gpr_predict<float>(
            val_type, nnt_id, krnl_id, (const float *)  param,
            n_train, pt_dim, (const float *)  X_train, ldX,
            (const float *)  Y_train, n_pred, (const float *)  X_pred, ldXp,
            (float *)  Y_pred, (float *)  stddev, NULL, NULL
        );
    }
}

template<typename VT>
static void exact_gpc_predict(
    const int val_type, const int nnt_id, const int krnl_id, const int n_class,
    const int n_sample, const VT *params, const int n_train, const int pt_dim, 
    const VT *X_train, const int ldX, const int *Y_train, const int n_pred, 
    const VT *X_pred, const int ldXp, int *Y_pred, VT *Y_pred_c, VT *probab
)
{
    VT *dnoises = NULL, *Ys = NULL;
    gpc_process_label(val_type, n_train, n_class, Y_train, (void **) &dnoises, (void **) &Ys);

    size_t VT_bytes = sizeof(VT);
    VT *stddev  = (VT *) malloc(VT_bytes * n_pred);
    VT *cov2    = (VT *) malloc(VT_bytes * n_pred * n_pred);
    VT *samples = (VT *) malloc(VT_bytes * n_pred * n_sample * n_class);
    VT *rndvec  = (VT *) malloc(VT_bytes * n_pred * n_sample);
    ASSERT_PRINTF(
        stddev != NULL && cov2 != NULL && samples != NULL && rndvec != NULL,
        "Failed to allocate work arrays for %s\n", __FUNCTION__
    );

    FILE *ouf = NULL;
    int dump_rndvec = 0;
    GET_ENV_INT_VAR(dump_rndvec, "EGPC_DUMP_RNDVEC", "dump_rndvec", 0, 0, 1);
    if (dump_rndvec)
    {
        srand48(19241112);  // Fixed seed for reproducibility
        char fname[64];
        sprintf(fname, "egpc_rndvec_%dx%dx%d.bin", n_class, n_pred, n_sample);
        ouf = fopen(fname, "wb");
    } else {
        srand48(time(NULL));
    }
    
    VT param[4] = {(VT) pt_dim, 0, 0, 0};
    for (int i_class = 0; i_class < n_class; i_class++)
    {
        VT *dnoise_i = dnoises  + i_class * n_train;
        VT *Y_i      = Ys       + i_class * n_train;
        VT *Yc_i     = Y_pred_c + i_class * n_pred;
        param[1] = params[i_class];
        param[2] = params[i_class + n_class];
        param[3] = params[i_class + n_class * 2];
        exact_gpr_predict<VT>(
            val_type, nnt_id, krnl_id, &param[0],
            n_train, pt_dim, X_train, ldX,
            Y_i, n_pred, X_pred, ldXp,
            Yc_i, stddev, cov2, dnoise_i
        );

        // cov2 = (cov2 + cov2') * 0.5;
        for (int i = 0; i < n_pred - 1; i++)
        {
            for (int j = i + 1; j < n_pred; j++)
            {
                int idx1 = j * n_pred + i;
                int idx2 = i * n_pred + j;
                VT val = 0.5 * (cov2[idx1] + cov2[idx2]);
                cov2[idx1] = val;
                cov2[idx2] = val;
            }
        }

        VT *samples_i = samples + i_class * n_pred * n_sample;
        VT *zero_vec = stddev;
        memset(zero_vec, 0, VT_bytes * n_pred);
        mvnrnd(val_type, n_pred, zero_vec, cov2, n_sample, rndvec);
        if (dump_rndvec) fwrite(rndvec, VT_bytes, n_pred * n_sample, ouf);
        #pragma omp parallel for schedule(static)
        for (int j = 0; j < n_sample; j++)
        {
            VT *samples_ji = samples_i + j * n_pred;
            VT *rndvec_j   = rndvec    + j * n_pred;
            for (int k = 0; k < n_pred; k++) samples_ji[k] = rndvec_j[k] + Yc_i[k];
        }
    }  // End of i_class loop

    if (dump_rndvec) fclose(ouf);

    gpc_pred_probab(
        val_type, n_class, n_pred, n_sample, (void *) samples,
        (const void *) Y_pred_c, Y_pred, (void *) probab
    );

    free(dnoises);
    free(Ys);
    free(stddev);
    free(cov2);
    free(samples);
    free(rndvec);
}

// GP classification prediction with a given kernel and its parameters using exact solve
void exact_gpc_predict(
    const int val_type, const int nnt_id, const int krnl_id, const int n_class,
    const int n_sample, const void *params, const int n_train, const int pt_dim, 
    const void *X_train, const int ldX, const int *Y_train, const int n_pred, 
    const void *X_pred, const int ldXp, int *Y_pred, void *Y_pred_c, void *probab
)
{
    if (val_type == VAL_TYPE_DOUBLE)
    {
        exact_gpc_predict<double>(
            val_type, nnt_id, krnl_id, n_class, 
            n_sample, (const double *) params, n_train, pt_dim, 
            (const double *) X_train, ldX, Y_train, n_pred, 
            (const double *) X_pred, ldXp, Y_pred, (double *) Y_pred_c, (double *) probab
        );
    }
    if (val_type == VAL_TYPE_FLOAT)
    {
        exact_gpc_predict<float>(
            val_type, nnt_id, krnl_id, n_class, 
            n_sample, (const float *)  params, n_train, pt_dim, 
            (const float *)  X_train, ldX, Y_train, n_pred, 
            (const float *)  X_pred, ldXp, Y_pred, (float *)  Y_pred_c, (float *)  probab
        );
    }
}
