#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <type_traits>
#include <omp.h>
#include <time.h>

#include "../cpu_linalg.hpp"
#include "../utils.h"
#include "precond_gp.h"
#include "nonneg_transform.h"
#include "gpc_common.h"

// Kernel-independent precomputation for preconditioned GP loss computation
void pgp_loss_init(
    const int val_type, const int nnt_id, const int n_train, const int pt_dim,
    const void *X_train, const int ldX, const void *Y_train, const int Y_val_type,
    const int npt_s, const int glr_rank, const int fsai_npt, const int n_iter, 
    const int n_vec, symm_kmat_alg_t kmat_alg, pgp_loss_p *pgp_loss
)
{
    pgp_loss_p pgp_loss_ = (pgp_loss_p) malloc(sizeof(pgp_loss_s));
    memset(pgp_loss_, 0, sizeof(pgp_loss_s));

    // 1. Copy parameters
    pgp_loss_->val_type = val_type;
    pgp_loss_->nnt_id   = nnt_id;
    pgp_loss_->n_train  = n_train;
    pgp_loss_->pt_dim   = pt_dim;
    pgp_loss_->npt_s    = npt_s;
    pgp_loss_->glr_rank = glr_rank;
    pgp_loss_->fsai_npt = fsai_npt;
    pgp_loss_->n_iter   = n_iter;
    pgp_loss_->n_vec    = n_vec;
    pgp_loss_->kmat_alg = kmat_alg;
    if (pt_dim > 3 && kmat_alg == SYMM_KMAT_ALG_H2_PROXY)
    {
        WARNING_PRINTF("Cannot use H2 matrix for pt_dim > 3, will use dense kernel matrix\n");
        pgp_loss_->kmat_alg = SYMM_KMAT_ALG_DENSE_FORM;
    }

    // 2. Copy training data
    size_t VT_bytes = (val_type == VAL_TYPE_DOUBLE) ? sizeof(double) : sizeof(float);
    size_t Y_VT_bytes = (Y_val_type == VAL_TYPE_DOUBLE) ? sizeof(double) : (Y_val_type == VAL_TYPE_FLOAT) ? sizeof(float) : sizeof(int);

    pgp_loss_->X_train = malloc(VT_bytes * n_train * pt_dim);
    pgp_loss_->Y_train = malloc(Y_VT_bytes * n_train);
    ASSERT_PRINTF(pgp_loss_->X_train != NULL && pgp_loss_->Y_train != NULL, "Failed to allocate work arrays for pgp_loss\n");
    // copy_matrix works on row-major matrices
    copy_matrix(VT_bytes, pt_dim, n_train, X_train, ldX, pgp_loss_->X_train, n_train, 1);
    memcpy(pgp_loss_->Y_train, Y_train, Y_VT_bytes * n_train);

    // 3. Set up octree and store it if we are using H2 matrix
    octree_p octree = NULL;
    if (pgp_loss_->kmat_alg == SYMM_KMAT_ALG_H2_PROXY)
    {
        int leaf_nmax = 400;
        double leaf_emax_d = 0;
        float  leaf_emax_f = 0;
        void *leaf_emax_ = NULL;
        if (val_type == VAL_TYPE_DOUBLE) leaf_emax_ = (void *) &leaf_emax_d;
        if (val_type == VAL_TYPE_FLOAT ) leaf_emax_ = (void *) &leaf_emax_f;
        octree_build(
            n_train, pt_dim, val_type, (const void *) X_train, 
            leaf_nmax, leaf_emax_, &octree
        );
    }
    pgp_loss_->octree = octree;

    *pgp_loss = pgp_loss_;
}

// Free an initialized pgp_loss struct
void pgp_loss_free(pgp_loss_p *pgp_loss)
{
    pgp_loss_p pgp_loss_ = *pgp_loss;
    if (pgp_loss_ == NULL) return;
    free(pgp_loss_->X_train);
    free(pgp_loss_->Y_train);
    free(pgp_loss_);
    *pgp_loss = NULL;
}

// Stochastic Lanczos quadrature approximation
// Input parameters:
//   val_type : Data type of coordinate and kernel values, 0 for double, 1 for float
//   kmat_alg : Algorithm and data structure for the square kernel matrix
//   dkmat    : Dense kernel matrix for the training data
//   ss_h2mat : H2 matrix for the training data
//   ap       : AFN preconditioner for the kernel matrix
//   n        : Number of training data
//   n_iter   : Number of Lanczos iterations
//   n_vec    : Number of random vectors
// Output parameters:
//   *val : Estimated tr(det(K))
//   dval : Size 3, estimated tr(K^{-1} * (d K / d {l, f, s}))
template<typename VT>
static void lanquad(
    const int val_type, const int kmat_alg, dense_krnl_mat_p dkmat, ss_h2mat_p ss_h2mat, 
    afn_precond_p ap, const int n, const int n_iter, const int n_vec, VT *val, VT *dval
)
{
    int n_grad = 3;

    // 1. Generate radamacher random vectors, compute K^{-1} * Z and the tridiagonal matrices
    int dump_Z = 0;
    GET_ENV_INT_VAR(dump_Z, "LANQUAD_DUMP_Z", "dump_Z", 0, 0, 1);
    if (dump_Z) srand(19241112);  // Fixed seed for reproducibility
    else srand(time(NULL));
    VT *Z   = (VT *) malloc(sizeof(VT) * n * n_vec);
    VT *iKZ = (VT *) malloc(sizeof(VT) * n * n_vec);
    VT *T   = (VT *) malloc(sizeof(VT) * n_vec * n_iter * n_iter);
    ASSERT_PRINTF(Z != NULL && iKZ != NULL && T != NULL, "Failed to allocate work arrays for %s\n", __FUNCTION__);
    for (int i = 0; i < n * n_vec; i++)
    {
        Z[i] = (VT) ((rand() & 1) * 2 - 1);
        iKZ[i] = 0.0;
    }
    if (dump_Z)
    {
        char fname[64];
        sprintf(fname, "lanquad_Z_%dx%d.bin", n, n_vec);
        FILE *ouf = fopen(fname, "wb");
        fwrite(Z, sizeof(VT), n * n_vec, ouf);
        fclose(ouf);
    }

    matmul_fptr A_mm = NULL;
    void *A_ptr = NULL;
    if (kmat_alg == SYMM_KMAT_ALG_DENSE || kmat_alg == SYMM_KMAT_ALG_DENSE_FORM)
    {
        A_mm  = (matmul_fptr) dense_krnl_mat_krnl_matmul;
        A_ptr = (void *) dkmat;
    }
    if (kmat_alg == SYMM_KMAT_ALG_H2_PROXY)
    {
        A_mm  = (matmul_fptr) ss_h2mat_krnl_matmul;
        A_ptr = (void *) ss_h2mat;
    }
    mpcg(
        n, n_vec, n_iter, val_type,
        A_mm, A_ptr, (matmul_fptr) afn_precond_apply, (void *) ap,
        Z, n, iKZ, n, (void *) T
    );

    // 2. Compute (d K / d {l, f, s}) * Z
    // col_idx = ((j - 1) * n_vec + 1) : (j * n_vec);
    // dKZ(:, col_idx) = dKfuns{j}(Z);
    VT *dKZ = (VT *) malloc(sizeof(VT) * n * (1 + n_grad) * n_vec);
    ASSERT_PRINTF(dKZ != NULL, "Failed to allocate work arrays for %s\n", __FUNCTION__);
    VT *K_Z    = dKZ + n * n_vec * 3;
    VT *dKdl_Z = dKZ;
    VT *dKdf_Z = dKZ + n * n_vec ;
    VT *dKds_Z = dKZ + n * n_vec * 2;
    if (kmat_alg == SYMM_KMAT_ALG_DENSE || kmat_alg == SYMM_KMAT_ALG_DENSE_FORM)
    {
        dense_krnl_mat_grad_matmul(
            dkmat, n_vec, Z, n, K_Z, 
            dKdl_Z, dKdf_Z, dKds_Z, n
        );
    }
    if (kmat_alg == SYMM_KMAT_ALG_H2_PROXY)
    {
        ss_h2mat_grad_matmul(
            ss_h2mat, n_vec, Z, n, K_Z,
            dKdl_Z, dKdf_Z, dKds_Z, n
        );
    }

    // 3. Apply preconditioner to Z
    // gt  = afn_grad_trace(ap);
    // iMZ = afn_apply(ap, Z);
    // dMZ = afn_dapply(ap, Z);
    VT *iMZ = (VT *) malloc(sizeof(VT) * n * n_vec);
    VT *dMZ = (VT *) malloc(sizeof(VT) * n * n_grad * n_vec);
    ASSERT_PRINTF(
        iMZ != NULL && dMZ != NULL,
        "Failed to allocate work arrays for %s\n", __FUNCTION__
    );
    afn_precond_grad_trace(ap);
    VT *gt = (VT *) ap->gt;
    afn_precond_apply(ap, n_vec, Z, n, iMZ, n);
    afn_precond_dapply(ap, n_vec, Z, n, dMZ, n);

    // 4. Compute vals and dvals
    // Query workspace size for xSYEV is annoying, let's overkill it
    int lwork = (3 + n_iter) * n_iter, info;
    VT *vals   = (VT *) malloc(sizeof(VT) * n_vec);
    VT *dvals  = (VT *) malloc(sizeof(VT) * n_grad * n_vec);
    VT *work   = (VT *) malloc(sizeof(VT) * lwork);
    VT *eigval = (VT *) malloc(sizeof(VT) * n_iter);
    ASSERT_PRINTF(
        vals != NULL && dvals != NULL && work != NULL && eigval != NULL,
        "Failed to allocate work arrays for %s\n", __FUNCTION__
    );
    for (int i = 0; i < n_vec; i++)
    {
        // [V, D] = eig(T{i});
        VT *T_i = T + i * n_iter * n_iter;
        int k = 0;
        for (k = 0; k < n_iter; k++) if (T_i[k * n_iter + k] == 0) break;
        xsyev_(vector, upper, &k, T_i, &n_iter, eigval, work, &lwork, &info);
        ASSERT_PRINTF(info == 0, "xSYEV returned error %d\n", info);

        // lam = abs(diag(D));
        // tau = V(1, :).^2;
        // vals(i) = tau * log(lam);
        vals[i] = 0.0;
        for (int j = 0; j < k; j++)
        {
            VT t1 = T_i[j * n_iter] * T_i[j * n_iter];
            VT t2 = std::log(std::abs(eigval[j]));
            vals[i] += t1 * t2;
        }

        VT *iKZ_i = iKZ + i * n;
        VT *iMZ_i = iMZ + i * n;
        for (int j = 0; j < n_grad; j++)
        {
            // dAZ_idx = (j - 1) * n_vec + i;
            // dvals(j, i) = iAZ(:, i)' * dAZ(:, dAZ_idx);
            int dKZ_idx = j * n_vec + i;
            VT *dKZ_j = dKZ + dKZ_idx * n;
            VT res = 0;
            #pragma omp simd
            for (int l = 0; l < n; l++) res += iKZ_i[l] * dKZ_j[l];
            int dvals_idx = j + i * n_grad;
            dvals[dvals_idx] = res;
            // dvals(j, i) = vt(j) + dvals(j, i) - iMZ(:, i)' * dMZ{j}(:, i);
            VT *dMZ_ji = dMZ + (j * n_vec + i) * n;
            res = 0;
            #pragma omp simd
            for (int l = 0; l < n; l++) res += iMZ_i[l] * dMZ_ji[l];
            dvals[dvals_idx] += gt[j] - res;
        }
    }  // End of i loop

    // 5. Compute the final val and dval
    *val = 0.0;
    memset(dval, 0, sizeof(VT) * n_grad);
    for (int j = 0; j < n_vec; j++)
    {
        *val += vals[j];
        for (int i = 0; i < n_grad; i++) dval[i] += dvals[i + j * n_grad];
    }
    *val = (*val) * (VT) n / (VT) n_vec;
    VT *ap_logdet = (VT *) ap->logdet;
    *val = *val + ap_logdet[0];
    for (int i = 0; i < n_grad; i++) dval[i] = dval[i] / (VT) n_vec;

    // Free work arrays and exit
    free(Z);
    free(iKZ);
    free(T);
    free(dKZ);
    free(iMZ);
    free(dMZ);
    free(vals);
    free(dvals);
    free(work);
    free(eigval);
}

template<typename VT>
static void precond_gpr_loss_compute(
    pgp_loss_p pgp_loss, const int krnl_id, const VT *param, 
    const VT *Y_train, const void *dnoise, VT *L, VT *L_grad
)
{
    int val_type = pgp_loss->val_type;
    int n_train  = pgp_loss->n_train;
    int n_iter   = pgp_loss->n_iter;
    int n_vec    = pgp_loss->n_vec;

    const int is_float  = std::is_same<VT, float>::value;
    const int is_double = std::is_same<VT, double>::value;

    // 1. Transform the hyper-parameters
    VT nnt_param[4], nnt_dparam[4];
    nnt_param[0] = param[0];
    for (int i = 1; i < 4; i++) 
        nonneg_transform(val_type, pgp_loss->nnt_id, &param[i], &nnt_param[i], &nnt_dparam[i]);
    
    // 2. Compute the kernel matrix and its derivatives w.r.t. hyper-parameters
    int  kmat_alg = pgp_loss->kmat_alg;
    void *X_train = pgp_loss->X_train;
    octree_p octree = pgp_loss->octree;
    dense_krnl_mat_p dkmat = NULL;
    ss_h2mat_p ss_h2mat = NULL;
    matmul_fptr A_mm = NULL;
    void *A_ptr = NULL;
    if (kmat_alg == SYMM_KMAT_ALG_DENSE || kmat_alg == SYMM_KMAT_ALG_DENSE_FORM)
    {
        dense_krnl_mat_init(
            n_train, n_train, X_train, n_train, n_train, X_train, 
            nnt_param, dnoise, krnl_id, val_type, &dkmat
        );
        A_mm  = (matmul_fptr) dense_krnl_mat_krnl_matmul;
        A_ptr = (void *) dkmat;
    }
    if (kmat_alg == SYMM_KMAT_ALG_H2_PROXY)
    {
        VT h2_reltol = 0;  // TODO: tune this parameter
        if (is_double) h2_reltol = 1e-10;
        if (is_float)  h2_reltol = 1e-5;
        ss_h2mat_init(octree, (void *) &nnt_param[0], dnoise, krnl_id, 1, (void *) &h2_reltol, &ss_h2mat);
        A_mm  = (matmul_fptr) ss_h2mat_krnl_matmul;
        A_ptr = (void *) ss_h2mat;
    }

    // 3. Build the AFN preconditioner
    afn_precond_p ap = NULL;
    int need_grad = 1;
    afn_precond_build(
        val_type, krnl_id, (void *) nnt_param, dnoise, 
        n_train, pgp_loss->pt_dim, pgp_loss->X_train, n_train,
        pgp_loss->npt_s, pgp_loss->glr_rank, pgp_loss->fsai_npt, 
        octree, need_grad, &ap
    );

    // We can try to populate the dense kernel matrix now since the 
    // memory-consuming AFN preconditioner has been built. The mpcg, 
    // lanquad, and the rest of this function only need some O(n) arrays. 
    if (kmat_alg == SYMM_KMAT_ALG_DENSE_FORM)
        dense_krnl_mat_populate(dkmat);

    // 4. First term in L and (d L / d {l, f, s})
    int nrhs = 1;
    VT *iKY = (VT *) malloc(sizeof(VT) * n_train);
    VT *vt  = (VT *) malloc(sizeof(VT) * n_train * 4);
    VT *T   = (VT *) malloc(sizeof(VT) * nrhs * n_iter * n_iter);
    memset(iKY, 0, sizeof(VT) * n_train);
    matmul_fptr invM_mm = (matmul_fptr) afn_precond_apply;
    void *invM_ptr = (void *) ap;
    mpcg(
        n_train, nrhs, n_iter, val_type, 
        A_mm, A_ptr, invM_mm, invM_ptr,
        (void *) Y_train, n_train, iKY, n_train, (void *) T
    );
    VT *K_iKY = vt, *dKdl_iKY = vt + n_train;
    VT *dKdf_iKY = vt + n_train * 2, *dKds_iKY = vt + n_train * 3;
    if (kmat_alg == SYMM_KMAT_ALG_DENSE || kmat_alg == SYMM_KMAT_ALG_DENSE_FORM)
    {
        dense_krnl_mat_grad_matmul(
            dkmat, nrhs, iKY, n_train, K_iKY, 
            dKdl_iKY, dKdf_iKY, dKds_iKY, n_train
        );
    }
    if (kmat_alg == SYMM_KMAT_ALG_H2_PROXY)
    {
        ss_h2mat_grad_matmul(
            ss_h2mat, nrhs, iKY, n_train, K_iKY,
            dKdl_iKY, dKdf_iKY, dKds_iKY, n_train
        );
    }
    VT L1 = 0, L1_grad[3] = {0, 0, 0};
    #pragma omp simd 
    for (int i = 0; i < n_train; i++)
    {
        L1 += Y_train[i] * iKY[i];
        L1_grad[0] += iKY[i] * dKdl_iKY[i];
        L1_grad[1] += iKY[i] * dKdf_iKY[i];
        L1_grad[2] += iKY[i] * dKds_iKY[i];
    }
    for (int i = 0; i < 3; i++) L1_grad[i] *= nnt_dparam[i + 1];

    // 5. Second term in L and (d L / d {l, f, s})
    VT L2 = 0, L2_grad[3] = {0, 0, 0};
    lanquad<VT>(val_type, kmat_alg, dkmat, ss_h2mat, ap, n_train, n_iter, n_vec, &L2, &L2_grad[0]);
    for (int i = 0; i < 3; i++) L2_grad[i] *= nnt_dparam[i + 1];

    // 6. Final loss and its gradient
    *L = 0.5 * (L1 + L2 + (VT) n_train * 1.837877066409345);
    for (int i = 0; i < 3; i++) L_grad[i] = 0.5 * (-L1_grad[i] + L2_grad[i]);

    afn_precond_free(&ap);
    dense_krnl_mat_free(&dkmat);
    free(iKY);
    free(vt);
    free(T);
}

// Compute the preconditioned GP loss and its derivatives w.r.t. l, f, and s
void precond_gpr_loss_compute(pgp_loss_p pgp_loss, const int krnl_id, const void *param, void *L, void *L_grad, void *dnoise)
{
    if (pgp_loss->val_type == VAL_TYPE_DOUBLE)
    {
        precond_gpr_loss_compute<double>(
            pgp_loss, krnl_id, (const double *) param, 
            (double *) pgp_loss->Y_train, dnoise, (double *) L, (double *) L_grad
        );
    }
    if (pgp_loss->val_type == VAL_TYPE_FLOAT)
    {
        precond_gpr_loss_compute<float>(
            pgp_loss, krnl_id, (const float *)  param, 
            (float *)  pgp_loss->Y_train, dnoise, (float *)  L, (float *)  L_grad
        );
    }
}

template<typename VT>
static void precond_gpc_loss_compute(
    pgp_loss_p pgp_loss, const int krnl_id, const int n_class,
    const VT *params, VT *L, VT *L_grads
)
{
    int val_type = pgp_loss->val_type;
    int n_train  = pgp_loss->n_train;

    VT *dnoises = NULL, *Ys = NULL;
    gpc_process_label(val_type, n_train, n_class, (int *) pgp_loss->Y_train, (void **) &dnoises, (void **) &Ys);

    VT inv_n = 1.0 / (VT) n_train;
    VT Li = 0, Li_grad[3] = {0, 0, 0};
    VT param[4] = {(VT) pgp_loss->pt_dim, 0, 0, 0};
    *L = 0.0;
    for (int i_class = 0; i_class < n_class; i_class++)
    {
        VT *dnoise_i = dnoises + i_class * n_train;
        VT *Y_i      = Ys      + i_class * n_train;
        param[1] = params[i_class];
        param[2] = params[i_class + n_class];
        param[3] = params[i_class + n_class * 2];
        precond_gpr_loss_compute<VT>(
            pgp_loss, krnl_id, &param[0], 
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

// Compute the preconditioned GP classification loss and its derivatives w.r.t. [l, f, s] 
void precond_gpc_loss_compute(
    pgp_loss_p pgp_loss, const int krnl_id, const int n_class,
    const void *params, void *L, void *L_grads
)
{
    if (pgp_loss->val_type == VAL_TYPE_DOUBLE)
        precond_gpc_loss_compute<double>(pgp_loss, krnl_id, n_class, (const double *) params, (double *) L, (double *) L_grads);
    if (pgp_loss->val_type == VAL_TYPE_FLOAT)
        precond_gpc_loss_compute<float> (pgp_loss, krnl_id, n_class, (const float *)  params, (float *)  L, (float *)  L_grads);
}

template<typename VT>
static void precond_gpr_predict(
    const int val_type, const int nnt_id, const int krnl_id, const VT *param, 
    const int n_train, const int pt_dim, const VT *X_train, const int ldXt, 
    const VT *Y_train, const int n_pred, const VT *X_pred, const int ldXp, 
    const int npt_s, const int glr_rank, const int fsai_npt, const int max_iter, 
    const VT *rel_tol, symm_kmat_alg_t K11_alg, VT *Y_pred, VT *stddev, 
    VT *cov2, VT *dnoise
)
{
    const int is_float  = std::is_same<VT, float>::value;
    const int is_double = std::is_same<VT, double>::value;

    // 1. Transform the hyper-parameters
    VT nnt_param[4], nnt_dparam[4];
    nnt_param[0] = param[0];
    for (int i = 1; i < 4; i++) 
        nonneg_transform(val_type, nnt_id, &param[i], &nnt_param[i], &nnt_dparam[i]);

    // 2. Compute three kernel matrices
    dense_krnl_mat_p K11_dk = NULL, K12 = NULL, K22 = NULL;
    octree_p K11_octree = NULL;
    ss_h2mat_p K11_h2 = NULL;
    // K11 = kernel(X_train, X_train, dnoise);
    if (K11_alg == SYMM_KMAT_ALG_DENSE || K11_alg == SYMM_KMAT_ALG_DENSE_FORM)
    {
        dense_krnl_mat_init(
            n_train, ldXt, X_train, n_train, ldXt, X_train,
            nnt_param, dnoise, krnl_id, val_type, &K11_dk
        );
    }
    if (K11_alg == SYMM_KMAT_ALG_H2_PROXY)
    {
        int leaf_nmax = 400;
        VT leaf_emax = 0;
        VT h2_reltol = 0;  // TODO: tune this parameter
        if (is_double) h2_reltol = 1e-10;
        if (is_float)  h2_reltol = 1e-5;
        octree_build(
            n_train, pt_dim, val_type, (const void *) X_train, 
            leaf_nmax, (const void *) &leaf_emax, &K11_octree
        );
        ss_h2mat_init(K11_octree, (void *) &nnt_param[0], dnoise, krnl_id, 1, (void *) &h2_reltol, &K11_h2);
    }
    // K12 = kernel(X_train, X_pred,  []);
    dense_krnl_mat_init(
        n_train, ldXt, X_train, n_pred, ldXp, X_pred,
        nnt_param, NULL, krnl_id, val_type, &K12
    );
    // K22 = kernel(X_pred,  X_pred,  zeros(n_pred, 1));
    dense_krnl_mat_init(
        n_pred, ldXp, X_pred, n_pred, ldXp, X_pred,
        nnt_param, NULL, krnl_id, val_type, &K22
    );

    // 3. Build AFN preconditioner for PCG solver
    int need_grad = 0;
    afn_precond_p ap = NULL;
    afn_precond_build(
        val_type, krnl_id, (void *) nnt_param, dnoise, 
        n_train, pt_dim, X_train, ldXt, 
        npt_s, glr_rank, fsai_npt, 
        K11_octree, need_grad, &ap
    );

    // We can try to populate the dense kernel matrix now since the 
    // memory-consuming AFN preconditioner has been built. The bpcg
    // and the rest of this function only need some O(n) arrays.
    if (K11_alg == SYMM_KMAT_ALG_DENSE_FORM)
        dense_krnl_mat_populate(K11_dk);

    // 4.1 iK11_Y = K11 \ Y_train;
    int i_one = 1;
    VT *iK11_Y = (VT *) malloc(sizeof(VT) * n_train);
    int *iters = (int *) malloc(sizeof(int) * n_pred);
    memset(iK11_Y, 0, sizeof(VT) * n_train);
    matmul_fptr A_mm = NULL;
    void *A_ptr = NULL;
    if (K11_alg == SYMM_KMAT_ALG_DENSE || K11_alg == SYMM_KMAT_ALG_DENSE_FORM)
    {
        A_mm  = (matmul_fptr) dense_krnl_mat_krnl_matmul;
        A_ptr = (void *) K11_dk;
    }
    if (K11_alg == SYMM_KMAT_ALG_H2_PROXY)
    {
        A_mm  = (matmul_fptr) ss_h2mat_krnl_matmul;
        A_ptr = (void *) K11_h2;
    }
    matmul_fptr invM_mm = (matmul_fptr) afn_precond_apply;
    void *invM_ptr = (void *) ap;
    bpcg(
        n_train, i_one, val_type, max_iter, rel_tol,
        A_mm, A_ptr, invM_mm, invM_ptr,
        Y_train, n_train, iK11_Y, n_train, iters
    );
    if (iters[0] == -1)
    {
        WARNING_PRINTF(
            "AFN-PCG failed to converge to reltol %.2e in %d iterations for iK11_Y\n", 
            rel_tol[0], max_iter
        );
    }
    // 4.2 Y_pred = K12' * iK11_Y;
    VT *K12_ = (VT *) malloc(sizeof(VT) * n_train * n_pred);
    ASSERT_PRINTF(K12_ != NULL, "Failed to allocate work arrary for %s\n", __FUNCTION__);
    dense_krnl_mat_grad_eval(K12, K12_, NULL, NULL, NULL);
    const VT v_zero = 0.0, v_one = 1.0;
    xgemv_(
        trans, &n_train, &n_pred, &v_one,
        K12_, &n_train, iK11_Y,
        &i_one, &v_zero, Y_pred, &i_one
    );

    // 5.1 iK11_K12 = K11 \ K12;
    VT *iK11_K12 = (VT *) malloc(sizeof(VT) * n_train * n_pred);
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n_train * n_pred; i++) iK11_K12[i] = 0;
    bpcg(
        n_train, n_pred, val_type, max_iter, rel_tol,
        A_mm, A_ptr, invM_mm, invM_ptr,
        K12_, n_train, iK11_K12, n_train, iters
    );
    for (int i = 0; i < n_pred; i++)
    {
        if (iters[i] >= 0) continue;
        WARNING_PRINTF(
            "AFN-PCG failed to converge to reltol %.2e in %d iterations for iK11_K12(:, %d)\n", 
            rel_tol[0], max_iter, i+1
        );
    }
    // (5.2) cov2 = K22 - K12' * iK11_K12;
    //       stddev = sqrt(abs(diag(cov2)));
    VT *K22_ = (VT *) malloc(sizeof(VT) * n_pred * n_pred);
    ASSERT_PRINTF(K22_ != NULL, "Failed to allocate work arrary for %s\n", __FUNCTION__);
    dense_krnl_mat_grad_eval(K22, K22_, NULL, NULL, NULL);
    const VT v_neg_one = -1.0;
    xgemm_(
        trans, notrans, &n_pred, &n_pred, &n_train, &v_neg_one,
        K12_, &n_train, iK11_K12, &n_train,
        &v_one, K22_, &n_pred
    );
    for (int i = 0; i < n_pred; i++) stddev[i] = std::sqrt(std::abs(K22_[i * n_pred + i]));

    if (cov2 != NULL) memcpy(cov2, K22_, sizeof(VT) * n_pred * n_pred);

    dense_krnl_mat_free(&K11_dk);  
    ss_h2mat_free(&K11_h2);
    octree_free(&K11_octree);
    dense_krnl_mat_free(&K12);
    dense_krnl_mat_free(&K22);
    free(iK11_Y);
    free(iK11_K12);
    free(K12_);
    free(K22_);
    free(iters);
}

// Preconditioned GP prediction with a given kernel and its parameters
void precond_gpr_predict(
    const int val_type, const int nnt_id, const int krnl_id, const void *param, 
    const int n_train, const int pt_dim, const void *X_train, const int ldXt, 
    const void *Y_train, const int n_pred, const void *X_pred, const int ldXp, 
    const int npt_s, const int glr_rank, const int fsai_npt, const int max_iter, 
    const void *rel_tol, symm_kmat_alg_t K11_alg, void *Y_pred, void *stddev
)
{
    if (val_type == VAL_TYPE_DOUBLE)
    {
        precond_gpr_predict<double>(
            val_type, nnt_id, krnl_id, (const double *) param,
            n_train, pt_dim, (const double *) X_train, ldXt,
            (const double *) Y_train, n_pred, (const double *) X_pred, ldXp,
            npt_s, glr_rank, fsai_npt, max_iter, 
            (const double *) rel_tol, K11_alg, (double *) Y_pred, (double *) stddev,
            NULL, NULL
        );
    }
    if (val_type == VAL_TYPE_FLOAT)
    {
        precond_gpr_predict<float>(
            val_type, nnt_id, krnl_id, (const float *)  param,
            n_train, pt_dim, (const float *)  X_train, ldXt,
            (const float *)  Y_train, n_pred, (const float *)  X_pred, ldXp,
            npt_s, glr_rank, fsai_npt, max_iter, 
            (const float *)  rel_tol, K11_alg, (float *)  Y_pred, (float *)  stddev,
            NULL, NULL
        );
    }
}

template<typename VT>
static void precond_gpc_predict(
    const int val_type, const int nnt_id, const int krnl_id, const int n_class, 
    const int n_sample, const VT *params, const int n_train, const int pt_dim, 
    const VT *X_train, const int ldXt, const int *Y_train, const int n_pred, 
    const VT *X_pred, const int ldXp, const int npt_s, const int glr_rank, 
    const int fsai_npt, const int max_iter, const VT *rel_tol, symm_kmat_alg_t K11_alg, 
    int *Y_pred, VT *Y_pred_c, VT *probab
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
    GET_ENV_INT_VAR(dump_rndvec, "PGPC_DUMP_RNDVEC", "dump_rndvec", 0, 0, 1);
    if (dump_rndvec)
    {
        srand48(19241112);  // Fixed seed for reproducibility
        char fname[64];
        sprintf(fname, "pgpc_rndvec_%dx%dx%d.bin", n_class, n_pred, n_sample);
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
        precond_gpr_predict<VT>(
            val_type, nnt_id, krnl_id, &param[0], 
            n_train, pt_dim, X_train, ldXt, 
            Y_i, n_pred, X_pred, ldXp, 
            npt_s, glr_rank, fsai_npt, max_iter, 
            rel_tol, K11_alg, Yc_i, stddev,
            cov2, dnoise_i
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

void precond_gpc_predict(
    const int val_type, const int nnt_id, const int krnl_id, const int n_class, 
    const int n_sample, const void *params, const int n_train, const int pt_dim, 
    const void *X_train, const int ldXt, const int *Y_train, const int n_pred, 
    const void *X_pred, const int ldXp, const int npt_s, const int glr_rank, 
    const int fsai_npt, const int max_iter, const void *rel_tol, symm_kmat_alg_t K11_alg, 
    int *Y_pred, void *Y_pred_c, void *probab
)
{
    if (val_type == VAL_TYPE_DOUBLE)
    {
        precond_gpc_predict<double>(
            val_type, nnt_id, krnl_id, n_class,
            n_sample, (const double *) params, n_train, pt_dim, 
            (const double *) X_train, ldXt, Y_train, n_pred, 
            (const double *) X_pred, ldXp, npt_s, glr_rank, 
            fsai_npt, max_iter, (const double *) rel_tol, K11_alg,
            Y_pred, (double *) Y_pred_c, (double *) probab
        );
    }
    if (val_type == VAL_TYPE_FLOAT)
    {
        precond_gpc_predict<float>(
            val_type, nnt_id, krnl_id, n_class,
            n_sample, (const float *)  params, n_train, pt_dim, 
            (const float *)  X_train, ldXt, Y_train, n_pred, 
            (const float *)  X_pred, ldXp, npt_s, glr_rank, 
            fsai_npt, max_iter, (const float *)  rel_tol, K11_alg,
            Y_pred, (float *)  Y_pred_c, (float *)  probab
        );
    }
}
