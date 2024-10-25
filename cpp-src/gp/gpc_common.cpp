#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <limits>
#include <algorithm>
#include <omp.h>

#include "../utils.h"
#include "../common.h"
#include "../cpu_linalg.hpp"
#include "gpc_common.h"

template<typename VT>
static void gpc_process_label(const int n_train, const int n_class, const int *Y_train, VT **dnoises_, VT **Ys_)
{
    int max_label = *std::max_element(Y_train, Y_train + n_train);
    if (max_label >= n_class)
    {
        ERROR_PRINTF("Classification labels should range from [0, n_class-1], where n_class = %d\n", n_class);
        return;
    }

    size_t VT_bytes = sizeof(VT);
    const VT alpha = 0.01;
    VT *alphas  = (VT *) malloc(VT_bytes * n_train * n_class);
    VT *Ys      = (VT *) malloc(VT_bytes * n_train * n_class);
    VT *dnoises = (VT *) malloc(VT_bytes * n_train * n_class);
    ASSERT_PRINTF(
        alphas != NULL && Ys != NULL && dnoises != NULL,
        "Failed to allocate work arrays for %s\n", __FUNCTION__
    );

    for (int i = 0; i < n_train * n_class; i++) alphas[i] = alpha;
    for (int i = 0; i < n_train; i++) alphas[Y_train[i] * n_train + i] += 1.0;
    for (int i = 0; i < n_train * n_class; i++)
    {
        dnoises[i] = std::log(1.0 / alphas[i] + 1.0);
        Ys[i] = std::log(alphas[i]) - 0.5 * dnoises[i];
    }
    
    *dnoises_ = dnoises;
    *Ys_      = Ys;
    free(alphas);
}

// Convert training labels to dnoises and RHS vectors
void gpc_process_label(
    const int val_type, const int n_train, const int n_class, 
    const int *Y_train, void **dnoises, void **Ys
)
{
    if (val_type == VAL_TYPE_DOUBLE)
        gpc_process_label<double>(n_train, n_class, Y_train, (double **) dnoises, (double **) Ys);
    if (val_type == VAL_TYPE_FLOAT)
        gpc_process_label<float> (n_train, n_class, Y_train, (float **)  dnoises, (float **)  Ys);
}

// Marsaglia polar method for generating normal distribution
// Input parameters:
//   val_type : Data type of mu, sigma, and x, 0 for double, 1 for float
//   len      : Length of x
//   mu       : Mean of the normal distribution
//   sigma    : Standard deviation of the normal distribution
// Output parameter:
//   *x : Normal distribution samples
template<typename VT>
static void gen_normal_distr(const int len, const VT *mu, const VT *sigma, VT *x)
{
    VT mu_ = *mu, sigma_ = *sigma;
    VT u1, u2, w, mult, x1, x2, w_eps = std::numeric_limits<VT>::epsilon() * 10.0;
    for (int i = 0; i < len - 1; i += 2)
    {
        do 
        {
            u1 = (VT) (drand48() * 2.0 - 1.0);
            u2 = (VT) (drand48() * 2.0 - 1.0);
            w  = u1 * u1 + u2 * u2;
        } while (w >= 1.0 || w <= w_eps);
        mult = std::sqrt((-2.0 * std::log(w)) / w);
        x1 = u1 * mult;
        x2 = u2 * mult;
        x[i]   = mu_ + sigma_ * x1;
        x[i+1] = mu_ + sigma_ * x2;
    }
    if (len % 2)
    {
        do 
        {
            u1 = (VT) (drand48() * 2.0 - 1.0);
            u2 = (VT) (drand48() * 2.0 - 1.0);
            w  = u1 * u1 + u2 * u2;
        } while (w >= 1.0 || w <= w_eps);
        mult = std::sqrt((-2.0 * std::log(w)) / w);
        x1 = u1 * mult;
        x[len - 1] = mu_ + sigma_ * x1;
    }
}

// Modified from https://homepages.inf.ed.ac.uk/imurray2/code/matlab_octave_missing/mvnrnd.m
template<typename VT>
static void mvnrnd(const int dim, const VT *mu, const VT *S, const int nvec, VT *X)
{
    size_t VT_bytes = sizeof(VT);
    VT *L = (VT *) malloc(VT_bytes * dim * dim);
    VT *V = (VT *) malloc(VT_bytes * dim * nvec);
    ASSERT_PRINTF(L != NULL && V != NULL, "Failed to allocate work arrays for %s\n", __FUNCTION__);

    // Copy the mean vector to all columns of X
    for (int i = 0; i < nvec; i++) memcpy(X + i * dim, mu, VT_bytes * dim);

    // L = chol(S, 'lower');
    int info = 0;
    memcpy(L, S, VT_bytes * dim * dim);
    xpotrf_(lower, &dim, L, &dim, &info);
    ASSERT_PRINTF(info == 0, "xPOTRF failed with info = %d\n", info);
    for (int j = 1; j < dim; j++)
        for (int i = 0; i < j - 1; i++) L[j * dim + i] = 0.0;

    // V = randn(dim, nvec);
    VT v_zero = 0.0, v_one = 1.0;
    gen_normal_distr<VT>(dim * nvec, &v_zero, &v_one, V);

    // X = L * V + X;
    xgemm_(
        notrans, notrans, &dim, &nvec, &dim, 
        &v_one, L, &dim, V, &dim, 
        &v_one, X, &dim
    );

    free(L);
    free(V);
}

// Draw nvec random dim-dimensional vectors from a multivariate Gaussian distribution
void mvnrnd(const int val_type, const int dim, const void *mu, const void *S, const int nvec, void *X)
{
    if (val_type == VAL_TYPE_DOUBLE)
        mvnrnd<double>(dim, (const double *) mu, (const double *) S, nvec, (double *) X);
    if (val_type == VAL_TYPE_FLOAT)
        mvnrnd<float> (dim, (const float *)  mu, (const float *)  S, nvec, (float *)  X);
}

template<typename VT>
static void gpc_pred_probab(
    const int n_class, const int n_pred, const int n_sample, 
    VT *samples, const VT *Y_pred_c, int *Y_pred, VT *probab
)
{
    size_t VT_bytes = sizeof(VT);

    for (int i = 0; i < n_pred; i++)
    {
        VT max_val = std::numeric_limits<VT>::lowest();
        for (int j = 0; j < n_class; j++)
        {
            if (Y_pred_c[i + j * n_pred] > max_val)
            {
                max_val = Y_pred_c[i + j * n_pred];
                Y_pred[i] = j;
            }
        }
    }

    #pragma omp parallel for
    for (int i = 0; i < n_pred * n_sample * n_class; i++) samples[i] = std::exp(samples[i]);
    for (int i = 0; i < n_pred * n_class; i++) probab[i] = 0.0;

    int n_thread = omp_get_num_threads();
    VT *workbuf = (VT *) malloc(VT_bytes * n_thread * n_class * 2);
    ASSERT_PRINTF(workbuf != NULL, "Failed to allocate work buffer for %s\n", __FUNCTION__);

    const VT inv_n_sample = 1.0 / n_sample;
    #pragma omp parallel num_threads(n_thread)
    {
        int tid = omp_get_thread_num();
        VT *work = workbuf + tid * n_class * 2;
        VT *samples_ij = work;
        VT *probab_i   = work + n_class;

        #pragma omp for schedule(static)
        for (int i = 0; i < n_pred; i++)
        {
            memset(probab_i, 0, VT_bytes * n_class);
            for (int j = 0; j < n_sample; j++)
            {
                int ij_idx = i + j * n_pred;
                VT sum_ij = 0.0;
                for (int k = 0; k < n_class; k++)
                {
                    samples_ij[k] = samples[ij_idx + k * n_pred * n_sample];
                    sum_ij += samples_ij[k];
                }
                sum_ij = 1.0 / sum_ij;
                for (int k = 0; k < n_class; k++) probab_i[k] += samples_ij[k] * sum_ij;
            }
            for (int k = 0; k < n_class; k++) probab[i + k * n_pred] = probab_i[k] * inv_n_sample;
        }  // End of i loop
    }  // End of "#pragma omp parallel"

    free(workbuf);
}

// Compute predicted labels and probabilities for GP classification
void gpc_pred_probab(
    const int val_type, const int n_class, const int n_pred, const int n_sample, 
    void *samples, const void *Y_pred_c, int *Y_pred, void *probab
)
{
    if (val_type == VAL_TYPE_DOUBLE)
    {
        gpc_pred_probab<double>(
            n_class, n_pred, n_sample, (double *) samples, 
            (const double *) Y_pred_c, Y_pred, (double *) probab
        );
    }
    if (val_type == VAL_TYPE_FLOAT)
    {
        gpc_pred_probab<float>(
            n_class, n_pred, n_sample, (float *)  samples, 
            (const float *)  Y_pred_c, Y_pred, (float *)  probab
        );
    }

}
