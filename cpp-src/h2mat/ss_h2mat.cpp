#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "../common.h"
#include "../kernels/kernels.h"
#include "ss_h2mat.h"
#include "h2mat_proxy_points.h"
#include "h2mat_build.h"
#include "h2mat_matmul.h"
#include "ss_h2mat.h"

// Initialize an ss_h2mat struct for computing 
// f^2 * (K(X, X, l) + s * I) and its derivative w.r.t. l, f, and s
void ss_h2mat_init(
    octree_p octree, const void *param, const void *dnoise, const int krnl_id, 
    const int algo, void *reltol, ss_h2mat_p *ss_h2mat
)
{
    ss_h2mat_p ss_h2mat_ = (ss_h2mat_p) malloc(sizeof(ss_h2mat_s));

    size_t val_bytes = (octree->val_type == VAL_TYPE_DOUBLE) ? sizeof(double) : sizeof(float);
    ss_h2mat_->param = malloc(4 * val_bytes);
    memcpy(ss_h2mat_->param, param, 4 * val_bytes);

    size_t dnoise_bytes = val_bytes * octree->npt;
    ss_h2mat_->dnoise = malloc(dnoise_bytes);
    memset(ss_h2mat_->dnoise, 0, dnoise_bytes);
    if (dnoise != NULL) memcpy(ss_h2mat_->dnoise, dnoise, dnoise_bytes);

    krnl_func krnl = NULL, gkrnl = NULL;
    get_krnl_grad_func(krnl_id, &krnl, &gkrnl, NULL);

    h2mat_p K_h2mat = NULL, dKdl_h2mat = NULL;
    if (algo == 1)
    {
        h2m_2dbuf_p *K_pp = NULL, *dKdl_pp = NULL;
        h2m_octree_proxy_points(octree, krnl,  param, reltol, &K_pp);
        h2m_octree_proxy_points(octree, gkrnl, param, reltol, &dKdl_pp);
        h2mat_build_with_proxy_points(octree, K_pp,     krnl, ss_h2mat_->param, reltol, &K_h2mat);
        h2mat_build_with_proxy_points(octree, dKdl_pp, gkrnl, ss_h2mat_->param, reltol, &dKdl_h2mat);
        for (int i = 0; i < octree->n_level; i++)
        {
            h2m_2dbuf_free(&K_pp[i]);
            h2m_2dbuf_free(&dKdl_pp[i]);
        }
        free(K_pp);
        free(dKdl_pp);
    }

    ss_h2mat_->K_h2mat    = K_h2mat;
    ss_h2mat_->dKdl_h2mat = dKdl_h2mat;
    *ss_h2mat = ss_h2mat_;
}

// Free an initialized ss_h2mat struct
void ss_h2mat_free(ss_h2mat_p *ss_h2mat)
{
    ss_h2mat_p ss_h2mat_ = *ss_h2mat;
    if (ss_h2mat_ == NULL) return;
    h2mat_free(&ss_h2mat_->K_h2mat);
    h2mat_free(&ss_h2mat_->dKdl_h2mat);
    free(ss_h2mat_->param);
    free(ss_h2mat_->dnoise);
    free(ss_h2mat_);
    *ss_h2mat = NULL;
}

template<typename VT>
static void ss_h2mat_grad_matmul(
    ss_h2mat_p ss_h2mat, const int B_ncol, VT *B, const int ldB, 
    VT *K_B, VT *dKdl_B, VT *dKdf_B, VT *dKds_B, const int ldC
)
{
    // 1. Compute K_B0 = k(X, Y, l) * B and dKdl_B0 = (dK / dl) * B
    h2mat_matmul(ss_h2mat->K_h2mat, B_ncol, (const void *) B, ldB, (void *) K_B, ldC);
    if (dKdl_B != NULL)
        h2mat_matmul(ss_h2mat->dKdl_h2mat, B_ncol, (const void *) B, ldB, (void *) dKdl_B, ldC);

    // 2. Compute dKdf_B = 2 * f * K_B0, K_B = f^2 * K_B0, dKdl_B = f^2 * dKdl_B0
    int nrow = ss_h2mat->K_h2mat->octree->npt;
    VT *param = (VT *) ss_h2mat->param;
    VT _2f = param[2] * 2;
    VT f2  = param[2] * param[2];
    for (int j = 0; j < B_ncol; j++)
    {
        VT *K_Bj    = K_B    + j * ldC;
        VT *dKdf_Bj = dKdf_B + j * ldC;
        VT *dKdl_Bj = dKdl_B + j * ldC;
        if (dKdf_B != NULL)
        {
            #pragma omp simd
            for (int i = 0; i < nrow; i++) dKdf_Bj[i] = _2f * K_Bj[i];
        }
        #pragma omp simd
        for (int i = 0; i < nrow; i++) K_Bj[i] *= f2;   
        if (dKdl_B != NULL)
        {
            #pragma omp simd
            for (int i = 0; i < nrow; i++) dKdl_Bj[i] *= f2;
        }
    }

    // 3. Add s * B and diag(dnoise) * B to K_B, and compute dKds * B = eye(n) * B
    VT s = param[3];
    VT *dnoise = (VT *) ss_h2mat->dnoise;
    for (int j = 0; j < B_ncol; j++)
    {
        VT *Bj      = B      + j * ldB;
        VT *K_Bj    = K_B    + j * ldC;
        VT *dKds_Bj = dKds_B + j * ldC;
        #pragma omp simd
        for (int i = 0; i < nrow; i++) K_Bj[i] += (s + dnoise[i]) * Bj[i];
        if (dKds_B != NULL) memcpy(dKds_Bj, Bj, nrow * sizeof(VT));
    }
}

// Compute C = M * B, M is the dense kernel matrix or its derivate matrices, 
// B is a dense input matrix, and C is a dense result matrix
void ss_h2mat_grad_matmul(
    ss_h2mat_p ss_h2mat, const int B_ncol, void *B, const int ldB, 
    void *K_B, void *dKdl_B, void *dKdf_B, void *dKds_B, const int ldC
)
{
    if (ss_h2mat == NULL) return;
    int val_type = ss_h2mat->K_h2mat->octree->val_type;
    if (val_type == VAL_TYPE_DOUBLE)
    {
        ss_h2mat_grad_matmul<double>(
            ss_h2mat, B_ncol, (double *) B, ldB,
            (double *) K_B, (double *) dKdl_B, (double *) dKdf_B, (double *) dKds_B, ldC
        );
    }
    if (val_type == VAL_TYPE_FLOAT)
    {
        ss_h2mat_grad_matmul<float>(
            ss_h2mat, B_ncol, (float *)  B, ldB,
            (float *)  K_B, (float *)  dKdl_B, (float *)  dKdf_B, (float *)  dKds_B, ldC
        );
    }
}

// Compute K * B only, parameters are the same as ss_h2mat_grad_matmul()
void ss_h2mat_krnl_matmul(
    ss_h2mat_p ss_h2mat, const int B_ncol, void *B, const int ldB, 
    void *K_B, const int ldC
)
{
    ss_h2mat_grad_matmul(ss_h2mat, B_ncol, B, ldB, K_B, NULL, NULL, NULL, ldC);
}
