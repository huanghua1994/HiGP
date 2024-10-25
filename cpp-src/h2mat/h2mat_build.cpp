#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include "omp.h"

#include "../common.h"
#include "../kernels/kernels.h"
#include "h2mat_build.h"
#include "h2mat_utils.h"
#include "id_ppqr.h"

// Compute the H2 matrix reduced near/far pairs recursively
// Input parameters:
//   octree : Point partitioning octree
//   node0  : Index of the first node
//   node1  : Index of the second node
// Output parameters:
//   near_pairs  : Integer vector for storing the near pairs
//   far_pairs   : Integer vector for storing the far pairs
//   min_far_lvl : Minimum level of far pairs
static void h2m_calc_near_far_pairs_recursive(
    octree_p octree, const int node0, const int node1,
    h2m_2dbuf_p near_pairs, h2m_2dbuf_p far_pairs, int *min_far_lvl
)
{
    int  val_type     = octree->val_type;
    int  pt_dim       = octree->pt_dim;
    int  max_children = 1 << pt_dim;
    int  *children    = octree->children;
    int  *n_children  = octree->n_children;
    int  *node_lvl    = octree->node_lvl;
    void *enbox       = octree->enbox;

    if (node0 == node1)
    {
        // Leaf node, noting to do
        int nc = n_children[node0];
        if (nc == 0) return;
        // Non-leaf node, check interaction between children nodes
        int *children_nodes = children + node0 * max_children;
        for (int i = 0; i < nc; i++)
        {
            int child_i = children_nodes[i];
            for (int j = i; j < nc; j++)
            {
                int child_j = children_nodes[j];
                h2m_calc_near_far_pairs_recursive(octree, child_i, child_j, near_pairs, far_pairs, min_far_lvl);
            }
        }
    } else {
        int nc0  = n_children[node0];
        int nc1  = n_children[node1];
        int lvl0 = node_lvl[node0];
        int lvl1 = node_lvl[node1];
        // 1. The current node pair is far pair, no further recursion is needed
        size_t val_bytes = (val_type == VAL_TYPE_DOUBLE) ? sizeof(double) : sizeof(float);
        char *enbox0 = (char *) enbox;
        char *enbox1 = (char *) enbox;
        enbox0 += node0 * 2 * pt_dim * val_bytes;
        enbox1 += node1 * 2 * pt_dim * val_bytes;
        if (h2m_is_admissible_enbox_pair(val_type, pt_dim, enbox0, enbox1))
        {
            h2m_2dbuf_ivec_push(far_pairs, node0);
            h2m_2dbuf_ivec_push(far_pairs, node1);
            int max_lvl_n01 = (lvl0 > lvl1) ? lvl0 : lvl1;
            if (*min_far_lvl > max_lvl_n01) *min_far_lvl = max_lvl_n01;
            return;
        }
        // 2. Two inadmissible leaf nodes
        if ((nc0 == 0) && (nc1 == 0))
        {
            h2m_2dbuf_ivec_push(near_pairs, node0);
            h2m_2dbuf_ivec_push(near_pairs, node1);
            return;
        }
        // 3. node0 is a leaf node, node1 is a non-leaf node, check node0 with node1's children
        if ((nc0 == 0) && (nc1 > 0))
        {
            int *node1_children = children + node1 * max_children;
            for (int j = 0; j < nc1; j++)
            {
                int child_j = node1_children[j];
                h2m_calc_near_far_pairs_recursive(octree, node0, child_j, near_pairs, far_pairs, min_far_lvl);
            }
            return;
        }
        // 4. node1 is a leaf node, node0 is a non-leaf node, check node1 with node0's children
        if ((nc0 > 0) && (nc1 == 0))
        {
            int *node0_children = children + node0 * max_children;
            for (int i = 0; i < nc0; i++)
            {
                int child_i = node0_children[i];
                h2m_calc_near_far_pairs_recursive(octree, child_i, node1, near_pairs, far_pairs, min_far_lvl);
            }
            return;
        }
        // 5. Both node0 and node1 are non-leaf nodes, check node0's children with node1's children
        if ((nc0 > 0) && (nc1 > 0))
        {
            int *node0_children = children + node0 * max_children;
            int *node1_children = children + node1 * max_children;
            for (int i = 0; i < nc0; i++)
            {
                int child_i = node0_children[i];
                for (int j = 0; j < nc1; j++)
                {
                    int child_j = node1_children[j];
                    h2m_calc_near_far_pairs_recursive(octree, child_i, child_j, near_pairs, far_pairs, min_far_lvl);
                }
            }
            return;
        }
    }  // End of "if (node0 == node1)"
}

// Compute the H2 matrix reduced near/far lists for each node
// Input parameter:
//   octree : Point partitioning octree
// Output parameters:
//   node_n_near : Size octree->n_node, number of near pairs for each node
//   node_near   : Size octree->n_node * octree->n_node, each row stores the near pairs for a node
//   node_n_far  : Size octree->n_node, number of far pairs for each node
//   node_far    : Size octree->n_node * octree->n_node, each row stores the far pairs for a node
//   min_far_lvl : The minimum level of far pairs
static void h2m_calc_near_far_lists(
    octree_p octree, int *node_n_near, int *node_near, 
    int *node_n_far, int *node_far, int *min_far_lvl
)
{
    // 1. Recursively compute all near and far pairs 
    int root_node = octree->lvl_nodes[octree->ln_displs[0]];
    h2m_2dbuf_p near_pairs = NULL, far_pairs = NULL;
    h2m_2dbuf_init(&near_pairs, sizeof(int), 1024, 1);
    h2m_2dbuf_init(&far_pairs,  sizeof(int), 1024, 1);
    h2m_2dbuf_ivec_set_size(near_pairs, 0);
    h2m_2dbuf_ivec_set_size(far_pairs,  0);
    *min_far_lvl = octree->n_level - 1;
    h2m_calc_near_far_pairs_recursive(
        octree, root_node, root_node, 
        near_pairs, far_pairs, min_far_lvl
    );

    // 2. Convert the symmetric near and far pairs to lists
    int n_node = octree->n_node;
    memset(node_n_near, 0, sizeof(int) * n_node);
    memset(node_n_far,  0, sizeof(int) * n_node);
    int n_near_pairs = h2m_2dbuf_ivec_get_size(near_pairs) / 2;
    for (int i = 0; i < n_near_pairs; i++)
    {
        int node0 = near_pairs->data_i[2 * i];
        int node1 = near_pairs->data_i[2 * i + 1];
        int idx0  = node_n_near[node0];
        int idx1  = node_n_near[node1];
        node_near[node0 * n_node + idx0] = node1;
        node_near[node1 * n_node + idx1] = node0;
        node_n_near[node0]++;
        node_n_near[node1]++;
    }
    int n_far_pairs = h2m_2dbuf_ivec_get_size(far_pairs) / 2;
    for (int i = 0; i < n_far_pairs; i++)
    {
        int node0 = far_pairs->data_i[2 * i];
        int node1 = far_pairs->data_i[2 * i + 1];
        int idx0  = node_n_far[node0];
        int idx1  = node_n_far[node1];
        node_far[node0 * n_node + idx0] = node1;
        node_far[node1 * n_node + idx1] = node0;
        node_n_far[node0]++;
        node_n_far[node1]++;
    }
    // Also put the diagonal blocks in the near list
    for (int i = 0; i < n_node; i++)
    {
        if (octree->n_children[i] > 0) continue;
        int idx = node_n_near[i];
        node_near[i * n_node + idx] = i;
        node_n_near[i]++;
    }
    h2m_2dbuf_free(&near_pairs);
    h2m_2dbuf_free(&far_pairs);
}

template <typename VT>
static void h2mat_build_VJ_proxy(h2mat_p h2mat, h2m_2dbuf_p *pp, const void *reltol)
{
    octree_p octree  = h2mat->octree;
    int npt          = octree->npt;
    int pt_dim       = octree->pt_dim;
    int n_level      = octree->n_level;
    int n_node       = octree->n_node;
    int val_type     = octree->val_type;
    int max_children = 1 << pt_dim;
    int *n_children  = octree->n_children;
    int *children    = octree->children;
    int *lvl_nnode   = octree->lvl_nnode;
    int *lvl_nodes   = octree->lvl_nodes;
    int *ln_displs   = octree->ln_displs;
    int *pt_cluster  = octree->pt_cluster;
    VT  *px          = (VT *) octree->px;
    VT  *enbox       = (VT *) octree->enbox;
    int min_far_lvl  = h2mat->min_far_lvl;
    krnl_func krnl   = h2mat->krnl;
    void *param      = h2mat->param;

    // 1. Allocate J_coords, V_mats, and working buffers
    h2m_2dbuf_p *J_coords = (h2m_2dbuf_p *) malloc(sizeof(h2m_2dbuf_p) * n_node);
    h2m_2dbuf_p *J_idxs   = (h2m_2dbuf_p *) malloc(sizeof(h2m_2dbuf_p) * n_node);
    h2m_2dbuf_p *V_mats   = (h2m_2dbuf_p *) malloc(sizeof(h2m_2dbuf_p) * n_node);
    for (int i = 0; i < n_node; i++)
    {
        h2m_2dbuf_init(J_coords + i, sizeof(VT),  0, pt_dim);
        h2m_2dbuf_init(J_idxs   + i, sizeof(int), 0, 1);
        h2m_2dbuf_init(V_mats   + i, sizeof(VT),  0, 0);
    }
    h2mat->J_coords = J_coords;
    h2mat->J_idxs   = J_idxs;
    h2mat->V_mats   = V_mats;
    const int thread_n_workbuf = 4;
    const int n_thread = omp_get_max_threads();
    h2m_2dbuf_p *thread_workbufs = (h2m_2dbuf_p *) malloc(sizeof(h2m_2dbuf_p) * thread_n_workbuf * n_thread);
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        for (int i = tid * thread_n_workbuf; i < (tid + 1) * thread_n_workbuf; i++)
            h2m_2dbuf_init(thread_workbufs + i, sizeof(VT), 1024, 1);
    }

    // 2. Hierarchical construction level by level
    for (int l = n_level - 1; l >= min_far_lvl; l--)
    {
        int *lvl_l_nodes = lvl_nodes + ln_displs[l];
        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            h2m_2dbuf_p *thread_workbuf = thread_workbufs + tid * thread_n_workbuf;

            // (1) Gather all skeleton points from children nodes
            #pragma omp for schedule(dynamic)
            for (int j = 0; j < lvl_nnode[l]; j++)
            {
                int node_j = lvl_l_nodes[j];
                int n_children_j = n_children[node_j];
                if (n_children_j == 0)
                {
                    // Leaf node: set skeleton points as all points in the enbox
                    int pt_s  = pt_cluster[2 * node_j];
                    int pt_e  = pt_cluster[2 * node_j + 1];
                    int npt_j = pt_e - pt_s + 1;  // [pt_s, pt_e], not [pt_s, pt_e)
                    h2m_2dbuf_resize(J_coords[node_j], sizeof(VT),  npt_j, pt_dim);
                    h2m_2dbuf_resize(J_idxs[node_j],   sizeof(int), npt_j, 1);
                    for (int k = 0; k < npt_j; k++) J_idxs[node_j]->data_i[k] = pt_s + k;
                    h2m_2dbuf_ivec_set_size(J_idxs[node_j], npt_j);
                    // copy_matrix works for row-major matrices
                    copy_matrix(sizeof(VT), pt_dim, npt_j, px + pt_s, npt, J_coords[node_j]->data, npt_j, 0);
                } else {
                    int *children_j = children + node_j * max_children;
                    int sum = 0;
                    for (int k = 0; k < n_children_j; k++) sum += J_coords[children_j[k]]->nrow;
                    h2m_2dbuf_resize(J_idxs[node_j],   sizeof(int), sum, 1);
                    h2m_2dbuf_resize(J_coords[node_j], sizeof(VT),  sum, pt_dim);
                    sum = 0;
                    for (int k = 0; k < n_children_j; k++) 
                    {
                        int child_k = children_j[k];
                        int npt_k   = J_coords[child_k]->nrow;
                        memcpy(J_idxs[node_j]->data_i + sum, J_idxs[child_k]->data_i, sizeof(int) * npt_k);
                        VT *src_k = (VT *) J_coords[child_k]->data;
                        VT *dst_k = ((VT *) J_coords[node_j]->data) + sum;
                        int lds = J_coords[child_k]->nrow;
                        int ldd = J_coords[node_j]->nrow;
                        copy_matrix(sizeof(VT), pt_dim, npt_k, src_k, lds, dst_k, ldd, 0);
                        sum += npt_k;
                    }
                    h2m_2dbuf_ivec_set_size(J_idxs[node_j], sum);
                }  // End of "if (n_children_j == 0)"
            }  // End of j loop

            #pragma omp barrier

            #pragma omp for schedule(dynamic)
            for (int j = 0; j < lvl_nnode[l]; j++)
            {
                int node_j = lvl_l_nodes[j];

                if (pp[l]->nrow == 0)
                {
                    // Fast path: no proxy point at this level
                    h2m_2dbuf_resize(V_mats[node_j], sizeof(VT), 1, J_coords[node_j]->nrow);
                    memset(V_mats[node_j]->data, 0, sizeof(VT) * J_coords[node_j]->nrow);
                    int skel_idx = 0;
                    h2m_2dbuf_gather_rows(VAL_TYPE_INT, J_idxs[node_j],   NULL, 1, &skel_idx);
                    h2m_2dbuf_gather_rows(val_type,     J_coords[node_j], NULL, 1, &skel_idx);
                    continue;
                }

                // (2) Shift current node's skeleton points to the origin
                h2m_2dbuf_p shift_coords_j = thread_workbuf[0];
                h2m_2dbuf_resize(shift_coords_j, sizeof(VT), J_coords[node_j]->nrow, J_coords[node_j]->ncol);
                VT *enbox_j = enbox + node_j * 2 * pt_dim;
                for (int k = 0; k < pt_dim; k++)
                {
                    VT center_k = enbox_j[k] + 0.5 * enbox_j[pt_dim + k];
                    VT *J_coords_jk     = (VT *) J_coords[node_j]->data;
                    VT *shift_coords_jk = (VT *) shift_coords_j->data;
                    J_coords_jk     += k * J_coords[node_j]->nrow;
                    shift_coords_jk += k * shift_coords_j->nrow;
                    #pragma omp simd
                    for (int i = 0; i < shift_coords_j->nrow; i++)
                        shift_coords_jk[i] = J_coords_jk[i] - center_k;
                }

                // (3) Build the kernel matrix K(pp[l], shift_coords_j)
                h2m_2dbuf_p K_blk = thread_workbuf[1];
                h2m_2dbuf_resize(K_blk, sizeof(VT), pp[l]->nrow, shift_coords_j->nrow);
                krnl(
                    pp[l]->nrow, pp[l]->nrow, pp[l]->data, 
                    shift_coords_j->nrow, shift_coords_j->nrow, shift_coords_j->data,
                    param, K_blk->nrow, K_blk->data, val_type
                );

                // (4) ID compression to get the column basis and skeleton points
                h2m_2dbuf_p QR_buff = thread_workbuf[2];
                h2m_2dbuf_p ID_buff = thread_workbuf[3];
                h2m_2dbuf_resize(QR_buff, sizeof(VT),  K_blk->nrow, K_blk->ncol);
                h2m_2dbuf_resize(ID_buff, sizeof(int), 4 * K_blk->ncol, 1);
                int rank = 0, max_rank = 0, id_nthread = 1, *skel_idx = NULL;
                void *V = NULL;
                id_ppqr(
                    K_blk->nrow, K_blk->ncol, val_type, K_blk->data, K_blk->nrow,
                    max_rank, reltol, id_nthread, &rank, &skel_idx, &V, ID_buff->data_i, QR_buff->data
                );
                h2m_2dbuf_resize(V_mats[node_j], sizeof(VT), rank, K_blk->ncol);
                memcpy(V_mats[node_j]->data, V, sizeof(VT) * rank * K_blk->ncol);
                h2m_2dbuf_gather_rows(VAL_TYPE_INT, J_idxs[node_j],   NULL, rank, skel_idx);
                h2m_2dbuf_gather_rows(val_type,     J_coords[node_j], NULL, rank, skel_idx);
                free(skel_idx);
                free(V);
                //DEBUG_PRINTF("Node %2d, initial / final rank = %d / %d\n", node_j, K_blk->ncol, rank);
            }  // End of j loop
        }  // End pf "#pragma omp parallel"
    }  // End of l loop

    // 3. Free thread working buffers
    for (int i = 0; i < thread_n_workbuf * n_thread; i++)
        h2m_2dbuf_free(thread_workbufs + i);
    free(thread_workbufs);
}

// Build an H2 matrix for K(X, X, l) with a given point partitioning 
// octree and a set of proxy points
void h2mat_build_with_proxy_points(
    octree_p octree, h2m_2dbuf_p *pp, krnl_func krnl,
    void *param, const void *reltol, h2mat_p *h2mat
)
{
    if (octree == NULL)
    {
        ERROR_PRINTF("Provided octree struct has not been initialized yet\n");
        *h2mat = NULL;
        return;
    }

    h2mat_p h2mat_ = (h2mat_p) malloc(sizeof(h2mat_s));
    memset(h2mat_, 0, sizeof(h2mat_s));

    // 1. Compute near and far lists
    int n_node = octree->n_node;
    int min_far_lvl = 0;
    int *node_n_near = (int *) malloc(sizeof(int) * n_node);
    int *node_n_far  = (int *) malloc(sizeof(int) * n_node);
    int *node_near   = (int *) malloc(sizeof(int) * n_node * n_node);
    int *node_far    = (int *) malloc(sizeof(int) * n_node * n_node);
    h2m_calc_near_far_lists(octree, node_n_near, node_near, node_n_far, node_far, &min_far_lvl);
    h2mat_->octree      = octree;
    h2mat_->n_node      = n_node;
    h2mat_->min_far_lvl = min_far_lvl;
    h2mat_->node_n_far  = node_n_far;
    h2mat_->node_far    = node_far;
    h2mat_->node_n_near = node_n_near;
    h2mat_->node_near   = node_near;
    h2mat_->krnl        = krnl;
    h2mat_->param       = param;
    
    // 2. Compute column basis matrices using proxy points
    if (octree->val_type == VAL_TYPE_DOUBLE)
        h2mat_build_VJ_proxy<double>(h2mat_, pp, reltol);
    if (octree->val_type == VAL_TYPE_FLOAT)
        h2mat_build_VJ_proxy<float> (h2mat_, pp, reltol);

    *h2mat = h2mat_;
}