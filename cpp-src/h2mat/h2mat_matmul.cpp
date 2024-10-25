#include <cstdio>
#include <cstdlib>
#include <cstring>
#include "omp.h"

#include "../common.h"
#include "../cpu_linalg.hpp"
#include "h2mat_utils.h"
#include "h2mat_matmul.h"

template<typename VT>
static void h2mat_matmul_fwd(h2mat_p h2mat, const int n, const VT *pX, const int ldpX, h2m_2dbuf_p *Y0)
{
    octree_p octree  = h2mat->octree;
    int pt_dim       = octree->pt_dim;
    int n_level      = octree->n_level;
    int max_children = 1 << pt_dim;
    int *n_children  = octree->n_children;
    int *children    = octree->children;
    int *lvl_nnode   = octree->lvl_nnode;
    int *lvl_nodes   = octree->lvl_nodes;
    int *ln_displs   = octree->ln_displs;
    int *pt_cluster  = octree->pt_cluster;
    int min_far_lvl  = h2mat->min_far_lvl;
    h2m_2dbuf_p *V_mats = h2mat->V_mats;

    const VT v_zero = 0.0, v_one = 1.0;

    for (int l = n_level - 1; l >= min_far_lvl; l--)
    {
        int *lvl_l_nodes = lvl_nodes + ln_displs[l];
        #pragma omp parallel for schedule(dynamic)
        for (int j = 0; j < lvl_nnode[l]; j++)
        {
            int node_j       = lvl_l_nodes[j];
            int n_children_j = n_children[node_j];
            h2m_2dbuf_p V_j  = V_mats[node_j];
            h2m_2dbuf_p Y0_j = Y0[node_j];
            h2m_2dbuf_resize(Y0_j, sizeof(VT), V_j->nrow, n);

            if (n_children_j == 0)
            {
                // Leaf node, directly multiply V_j * X_j
                const VT *pX_j = pX + pt_cluster[2 * node_j];
                xgemm_(
                    notrans, notrans, &V_j->nrow, &n, &V_j->ncol,
                    &v_one, (VT *) V_j->data, &V_j->nrow, pX_j, &ldpX,
                    &v_zero, (VT *) Y0_j->data, &Y0_j->nrow
                );
            } else {
                // Non-leaf node, gather Y0 from children node and multiply V_j * vertcat(Y0{children_j})
                int *children_j = children + node_j * max_children;
                VT *V_j_ptr = (VT *) V_j->data;
                for (int k = 0; k < n_children_j; k++)
                {
                    int child_k = children_j[k];
                    h2m_2dbuf_p Y0_k = Y0[child_k];
                    VT beta = (k == 0) ? 0.0 : 1.0;
                    xgemm_(
                        notrans, notrans, &V_j->nrow, &n, &Y0_k->nrow,
                        &v_one, V_j_ptr, &V_j->nrow, (VT *) Y0_k->data, &Y0_k->nrow,
                        &beta, (VT *) Y0_j->data, &Y0_j->nrow
                    );
                    V_j_ptr += V_j->nrow * Y0_k->nrow;
                }  // End of k loop
            }  // End of "if (n_children_j == 0)"
        }  // End of j loop
    }  // End of l loop
}

template<typename VT>
static void h2mat_matmul_mid(
    h2mat_p h2mat, const int n, const VT *pX, const int ldpX, 
    VT *pY, const int ldpY, h2m_2dbuf_p *Y0, h2m_2dbuf_p *Y1
)
{
    octree_p octree = h2mat->octree;
    int npt         = octree->npt;
    int n_node      = octree->n_node;
    int val_type    = octree->val_type;
    int *node_lvl   = octree->node_lvl;
    int *pt_cluster = octree->pt_cluster;
    int *node_n_far = h2mat->node_n_far;
    int *node_far   = h2mat->node_far;
    VT  *px         = (VT *) octree->px;
    h2m_2dbuf_p *J_coords = h2mat->J_coords;
    krnl_func krnl = h2mat->krnl;
    void *param = h2mat->param;

    const VT v_one = 1.0;

    #pragma omp parallel
    {
        h2m_2dbuf_p Bij = NULL;
        h2m_2dbuf_init(&Bij, sizeof(VT), 0, 0);

        #pragma omp for schedule(dynamic)
        for (int node0 = 0; node0 < n_node; node0++)
        {
            int node0_n_far = node_n_far[node0];
            if (node0_n_far == 0) continue;

            int lvl0 = node_lvl[node0];
            int *node0_far = node_far + node0 * n_node;

            // We only compute Y1[node0] += Bij * Y0[node1], 
            // Y1[node1] += Bji * Y0[node0] is computed in separately
            h2m_2dbuf_p Y1_i = Y1[node0];
            h2m_2dbuf_p J_coords_i = J_coords[node0];
            int Y1_i_nrow = J_coords_i->nrow;
            h2m_2dbuf_resize(Y1_i, sizeof(VT), Y1_i_nrow, n);
            memset(Y1_i->data, 0, sizeof(VT) * Y1_i->nrow * Y1_i->ncol);

            for (int j = 0; j < node0_n_far; j++)
            {
                int node1 = node0_far[j];
                int lvl1  = node_lvl[node1];
                
                // (1) node0 and node1 are on the same level, compress on both sides
                if (lvl0 == lvl1)
                {
                    h2m_2dbuf_p Y0_j = Y0[node1];
                    h2m_2dbuf_p J_coords_j = J_coords[node1];
                    h2m_2dbuf_resize(Bij, sizeof(VT), J_coords_i->nrow, J_coords_j->nrow);
                    krnl(
                        J_coords_i->nrow, J_coords_i->nrow, J_coords_i->data,
                        J_coords_j->nrow, J_coords_j->nrow, J_coords_j->data,
                        param, Bij->nrow, Bij->data, val_type
                    );
                    xgemm_(
                        notrans, notrans, &Y1_i_nrow, &n, &Y0_j->nrow,
                        &v_one, (VT *) Bij->data, &Bij->nrow, (VT *) Y0_j->data, &Y0_j->nrow,
                        &v_one, (VT *) Y1_i->data, &Y1_i->nrow
                    );
                }

                // (2) node1 is a leaf node and higher than node0, only compress on node0's side
                if (lvl0 > lvl1)
                {
                    int clu_s1 = pt_cluster[2 * node1];
                    int clu_n1 = pt_cluster[2 * node1 + 1] - clu_s1 + 1;
                    const VT *pX_ptr1 = pX + clu_s1;
                    h2m_2dbuf_resize(Bij, sizeof(VT), J_coords_i->nrow, clu_n1);
                    krnl(
                        J_coords_i->nrow, J_coords_i->nrow, J_coords_i->data,
                        clu_n1, npt, px + clu_s1,
                        param, Bij->nrow, Bij->data, val_type
                    );
                    xgemm_(
                        notrans, notrans, &Y1_i_nrow, &n, &clu_n1,
                        &v_one, (VT *) Bij->data, &Bij->nrow, pX_ptr1, &ldpX,
                        &v_one, (VT *) Y1_i->data, &Y1_i->nrow
                    );
                }

                // (3) node0 is a leaf node and higher than node1, only compress on node1's side
                if (lvl0 < lvl1)
                {
                    int clu_s0 = pt_cluster[2 * node0];
                    int clu_n0 = pt_cluster[2 * node0 + 1] - clu_s0 + 1;
                    VT *pY_ptr0 = pY + clu_s0;
                    h2m_2dbuf_p Y0_j = Y0[node1];
                    h2m_2dbuf_p J_coords_j = J_coords[node1];
                    h2m_2dbuf_resize(Bij, sizeof(VT), clu_n0, J_coords_j->nrow);
                    krnl(
                        clu_n0, npt, px + clu_s0, 
                        J_coords_j->nrow, J_coords_j->nrow, J_coords_j->data,
                        param, Bij->nrow, Bij->data, val_type
                    );
                    xgemm_(  \
                        notrans, notrans, &clu_n0, &n, &Y0_j->nrow,
                        &v_one, (VT *) Bij->data, &Bij->nrow, (VT *) Y0_j->data, &Y0_j->nrow,
                        &v_one, (VT *) pY_ptr0, &ldpY
                    );
                }
            }  // End of j loop
        }  // End of node0 loop

        h2m_2dbuf_free(&Bij);
    }  // End of "#pragma omp parallel"
}

template<typename VT>
static void h2mat_matmul_bwd(h2mat_p h2mat, const int n, h2m_2dbuf_p *Y1, VT *pY, const int ldpY)
{
    octree_p octree  = h2mat->octree;
    int pt_dim       = octree->pt_dim;
    int n_level      = octree->n_level;
    int max_children = 1 << pt_dim;
    int *n_children  = octree->n_children;
    int *children    = octree->children;
    int *lvl_nnode   = octree->lvl_nnode;
    int *lvl_nodes   = octree->lvl_nodes;
    int *ln_displs   = octree->ln_displs;
    int *pt_cluster  = octree->pt_cluster;
    int min_far_lvl  = h2mat->min_far_lvl;
    h2m_2dbuf_p *V_mats = h2mat->V_mats;

    const VT v_zero = 0.0, v_one = 1.0;

    int n_thread = omp_get_max_threads();
    h2m_2dbuf_p *thread_bufs = (h2m_2dbuf_p *) malloc(sizeof(h2m_2dbuf_p) * n_thread);
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        h2m_2dbuf_init(&thread_bufs[tid], sizeof(VT), 0, 0);
    }

    for (int l = min_far_lvl; l < n_level; l++)
    {
        int *lvl_l_nodes = lvl_nodes + ln_displs[l];
        #pragma omp parallel 
        {
            int tid = omp_get_thread_num();
            h2m_2dbuf_p tmp_Y = thread_bufs[tid];

            #pragma omp for schedule(dynamic)
            for (int j = 0; j < lvl_nnode[l]; j++)
            {
                int node_j = lvl_l_nodes[j];
                if (Y1[node_j]->nrow == 0) continue;

                // tmp_Y = V_j^T * Y1_j
                h2m_2dbuf_p V_j = V_mats[node_j];
                h2m_2dbuf_p Y1_j = Y1[node_j];
                h2m_2dbuf_resize(tmp_Y, sizeof(VT), V_j->ncol, n);
                xgemm_(
                    trans, notrans, &tmp_Y->nrow, &n, &V_j->nrow,
                    &v_one, (VT *) V_j->data, &V_j->nrow, (VT *) Y1_j->data, &Y1_j->nrow,
                    &v_zero, (VT *) tmp_Y->data, &tmp_Y->nrow
                );
                h2m_2dbuf_resize(Y1_j, sizeof(VT), tmp_Y->nrow, tmp_Y->ncol);
                memcpy(Y1_j->data, tmp_Y->data, sizeof(VT) * tmp_Y->nrow * tmp_Y->ncol);

                int n_children_j = n_children[node_j];
                if (n_children_j == 0)
                {
                    // Leaf node, accumulate tmp_Y to pY
                    int clu_s = pt_cluster[2 * node_j];
                    for (int v = 0; v < n; v++)
                    {
                        VT *pY_v = pY + clu_s + v * ldpY;
                        VT *tmp_Y_v = ((VT *) tmp_Y->data) + v * tmp_Y->nrow;
                        #pragma omp simd
                        for (int i = 0; i < tmp_Y->nrow; i++) pY_v[i] += tmp_Y_v[i];
                    }
                } else {
                    // Non-leaf node, push down tmp_Y to children
                    int *children_j = children + max_children * node_j;
                    int tmp_Y_srow = 0;
                    for (int k = 0; k < n_children_j; k++)
                    {
                        int child_k = children_j[k];
                        int child_k_nrow = V_mats[child_k]->nrow;
                        h2m_2dbuf_p Y1_k = Y1[child_k];
                        if (Y1_k->nrow == 0) 
                        {
                            h2m_2dbuf_resize(Y1_k, sizeof(VT), child_k_nrow, n);
                            VT *src = ((VT *) tmp_Y->data) + tmp_Y_srow;
                            copy_matrix(sizeof(VT), n, child_k_nrow, src, tmp_Y->nrow, Y1_k->data, Y1_k->nrow, 0);
                        } else {
                            for (int v = 0; v < n; v++)
                            {
                                VT *Y1_k_v = ((VT *) Y1_k->data) + v * Y1_k->nrow;
                                VT *tmp_Y_v = ((VT *) tmp_Y->data) + tmp_Y_srow + v * tmp_Y->nrow;
                                #pragma omp simd
                                for (int i = 0; i < child_k_nrow; i++) Y1_k_v[i] += tmp_Y_v[i];
                            }
                        }
                        tmp_Y_srow += child_k_nrow;
                    }  // End of k loop
                }  // End of "if (n_children_j == 0)"
            }  // End of j loop
        }  // End of "#pragma omp parallel"
    }  // End of l loop

    for (int i = 0; i < n_thread; i++)
        h2m_2dbuf_free(&thread_bufs[i]);
    free(thread_bufs);    
}

template<typename VT>
static void h2mat_matmul_dense(h2mat_p h2mat, const int n, const VT *pX, const int ldpX, VT *pY, const int ldpY)
{
    octree_p octree  = h2mat->octree;
    int npt          = octree->npt;
    int n_node       = octree->n_node;
    int val_type     = octree->val_type;
    int *pt_cluster  = octree->pt_cluster;
    int *node_n_near = h2mat->node_n_near;
    int *node_near   = h2mat->node_near;
    VT  *px          = (VT *) octree->px;
    krnl_func krnl = h2mat->krnl;
    void *param = h2mat->param;

    const VT v_one = 1.0;

    #pragma omp parallel
    {
        h2m_2dbuf_p Dij = NULL;
        h2m_2dbuf_init(&Dij, sizeof(VT), 0, 0);

        #pragma omp for schedule(dynamic)
        for (int node0 = 0; node0 < n_node; node0++)
        {
            int node0_n_near = node_n_near[node0];
            if (node0_n_near == 0) continue;
            int *node0_near = node_near + node0 * n_node;

            // We only compute pY_i += Dij * pX_j, 
            // pY_j += Dji * pX_i is computed in separately
            int clu_s0 = pt_cluster[2 * node0];
            int clu_n0 = pt_cluster[2 * node0 + 1] - clu_s0 + 1;
            VT  *pY_ptr0 = pY + clu_s0;

            for (int j = 0; j < node0_n_near; j++)
            {
                int node1 = node0_near[j];
                int clu_s1 = pt_cluster[2 * node1];
                int clu_n1 = pt_cluster[2 * node1 + 1] - clu_s1 + 1;
                const VT *pX_ptr1 = pX + clu_s1;

                h2m_2dbuf_resize(Dij, sizeof(VT), clu_n0, clu_n1);
                krnl(
                    clu_n0, npt, px + clu_s0, clu_n1, npt, px + clu_s1,
                    param, Dij->nrow, Dij->data, val_type
                );
                xgemm_(
                    notrans, notrans, &clu_n0, &n, &clu_n1,
                    &v_one, (VT *) Dij->data, &Dij->nrow, (VT *) pX_ptr1, &ldpX,
                    &v_one, (VT *) pY_ptr0, &ldpY
                );
            }  // End of j loop
        }  // End of node0 loop

        h2m_2dbuf_free(&Dij);
    }  // End of "#pragma omp parallel"
}

template<typename VT>
static void h2mat_matmul(
    h2mat_p h2mat, const int n, const VT *X, const int ldX, 
    VT *Y, const int ldY
)
{
    octree_p octree = h2mat->octree;
    int npt         = octree->npt;
    int n_node      = octree->n_node;
    int *fwd_perm   = octree->fwd_perm;
    int *bwd_perm   = octree->bwd_perm;

    // 1. Allocate working buffers
    const int ldpX = npt, ldpY = npt;
    VT *pX = (VT *) malloc(sizeof(VT) * npt * n);
    VT *pY = (VT *) malloc(sizeof(VT) * npt * n);
    ASSERT_PRINTF(
        pX != NULL && pY != NULL, 
        "Failed to allocate working buffers for %s\n", __FUNCTION__
    );
    h2m_2dbuf_p *Y0 = (h2m_2dbuf_p *) malloc(sizeof(h2m_2dbuf_p) * n_node);
    h2m_2dbuf_p *Y1 = (h2m_2dbuf_p *) malloc(sizeof(h2m_2dbuf_p) * n_node);
    for (int i = 0; i < n_node; i++)
    {
        h2m_2dbuf_init(Y0 + i, sizeof(VT), 0, n);
        h2m_2dbuf_init(Y1 + i, sizeof(VT), 0, n);
    }
    
    // 2. Permute the input matrix X to the order of the octree
    #pragma omp parallel
    {
        for (int j = 0; j < n; j++)
        {
            const VT *X_j = X + j * ldX;
            VT *pX_j = pX + j * ldpX;
            #pragma omp for schedule(static)
            for (int i = 0; i < npt; i++) pX_j[i] = X_j[bwd_perm[i]];
        }
    }
    memset(pY, 0, sizeof(VT) * npt * n);

    // 3. Forward pass (upward sweep) for V * Y
    h2mat_matmul_fwd<VT>(h2mat, n, pX, ldpX, Y0);

    // 4. Intermediate multiplication for B * (V * Y)
    h2mat_matmul_mid<VT>(h2mat, n, pX, ldpX, pY, ldpY, Y0, Y1);

    // 5. Backward pass (downward sweep) for V^T * (B * V * Y)
    h2mat_matmul_bwd<VT>(h2mat, n, Y1, pY, ldpY);

    // 6. Dense multiplication for D * Y
    h2mat_matmul_dense<VT>(h2mat, n, pX, ldpX, pY, ldpY);

    // 7. Permute the output matrix Y back to the original order
    #pragma omp parallel
    {
        for (int j = 0; j < n; j++)
        {
            VT *pY_j = pY + j * ldpY;
            VT *Y_j  = Y  + j * ldY;
            #pragma omp for schedule(static)
            for (int i = 0; i < npt; i++) Y_j[i] = pY_j[fwd_perm[i]];
        }
    }

    // 8. Free working buffers
    free(pX);
    free(pY);
    for (int i = 0; i < n_node; i++)
    {
        h2m_2dbuf_free(Y0 + i);
        h2m_2dbuf_free(Y1 + i);
    }
    free(Y0);
    free(Y1);
}

// Compute Y := K(X, X, l) * X
void h2mat_matmul(
    h2mat_p h2mat, const int n, const void *X, const int ldX, 
    void *Y, const int ldY
)
{
    if (h2mat == NULL)
    {
        ERROR_PRINTF("Uninitialized h2mat struct\n");
        return;
    }
    if (h2mat->octree->val_type == VAL_TYPE_DOUBLE)
        h2mat_matmul<double>(h2mat, n, (const double *) X, ldX, (double *) Y, ldY);
    if (h2mat->octree->val_type == VAL_TYPE_FLOAT)
        h2mat_matmul<float> (h2mat, n, (const float *)  X, ldX, (float *)  Y, ldY);
}
