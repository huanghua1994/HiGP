#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cassert>
#include <algorithm>

#include "octree.h"
#include "../common.h"
#include "../utils.h"

#define OCTREE_MAX_DIM      3
#define OCTREE_MAX_CHILDREN (1 << OCTREE_MAX_DIM)

// Extend the enclosing box a little bit to avoid numerical issues
// The original enclosing box and the extended enclosing box have the same center
template<typename VT>
static void extend_enbox(const int pt_dim, VT *enbox)
{
    for (int i = 0; i < pt_dim; i++)
    {
        VT ext_size = enbox[pt_dim + i] * 1e-5;
        enbox[i] -= ext_size;
        enbox[pt_dim + i] += 2.0 * ext_size;
    }
}

// Calculate the minimal enclosing box of a point set
// Input parameters:
//   npt    : Number of points in the point set
//   pt_dim : Dimension of the points
//   x      : Size ldx * pt_dim, col-major matrix, each row in x is a point coordinate
//   ldx    : Leading dimension of x, >= npt
// Output parameter:
//   enbox : Size 2 * pt_dim, the first pt_dim values are the lower corner, the 
//           second pt_dim values are the sizes of the enclosing box
template<typename VT>
static void calc_min_enbox(const int npt, const int pt_dim, const VT *x, const int ldx, VT *enbox)
{
    for (int i = 0; i < pt_dim; i++)
    {
        const VT *x_i = x + i * ldx;
        VT max_c = *std::max_element(x_i, x_i + npt);
        VT min_c = *std::min_element(x_i, x_i + npt);
        enbox[i] = min_c;
        enbox[pt_dim + i] = max_c - min_c;
    }
    extend_enbox(pt_dim, enbox);
}

// Recursive partitioning of the point
// Input parameters:
//   npt        : Number of points in the point set
//   pt_dim     : Dimension of the points
//   leaf_nmax  : Maximum number of points in a leaf node
//   leaf_emax  : Maximum size of the enclosing box of a leaf node
//   x_enbox    : If the enclosing box of the point set is already specified. 
//                If NULL, the enclosing box will be calculated from the point set
//   ldx        : Leading dimension of x and x_tmp, should always == global npt
//   x0idx      : Global index of the 1st point in x
//   my_level   : Level of the current node (root = 0)
//   workv      : Size global npt * pt_dim, VT work buffer
//   worki      : Size global npt * 3, int work buffer
// Input and output parameters:
//   x          : Size ldx * pt_dim, col-major matrix, each row in x is a point coordinate
//   pidx       : Size npt, the original point index of each point in x
//   node_lvl   : Size global npt, level of each node, root node at level 0
//   parent     : Size global npt, parent node index of each node
//   children   : Size global npt * (2^pt_dim), row-major, each row contains children indices of each node
//   n_children : Size global npt, number of children of each node
//   enbox      : Size global npt * (2 * pt_dim), each row contains the enclosing box of a node
//   pt_cluster : Size global npt * 2, start and end (included) indices of permuted points in each node
//   node_npt   : Size global npt, number of points in each node
//   node_idx   : Current number of nodes assigned in the whole tree
template<typename VT>
static int octree_recursive_partition(
    const int npt, const int pt_dim, const int leaf_nmax, const VT leaf_emax,
    const VT *x_enbox, VT *x, const int ldx, const int x0idx, int *pidx, 
    const int my_level, int *node_lvl, int *parent, int *children, int *n_children, 
    VT *enbox, int *pt_cluster, int *node_npt, int *node_idx, VT *workv, int *worki
)
{
    int max_children = 1 << pt_dim;

    // 1. Calculate the enclosing box of the point set
    VT curr_enbox[2 * OCTREE_MAX_DIM], curr_center[OCTREE_MAX_DIM];
    if (x_enbox == NULL) calc_min_enbox<VT>(npt, pt_dim, x, ldx, &curr_enbox[0]);
    else memcpy(&curr_enbox[0], x_enbox, sizeof(VT) * 2 * pt_dim);
    VT *curr_enbox_size = &curr_enbox[pt_dim];
    for (int i = 0; i < pt_dim; i++)
        curr_center[i] = curr_enbox[i] + curr_enbox_size[i] * 0.5;

    // 2. Check if the number of points or the size of the enclosing box is small enough
    VT max_enbox_size = *std::max_element(curr_enbox_size, curr_enbox_size + pt_dim);
    if ((npt <= leaf_nmax) || (max_enbox_size <= leaf_emax))
    {
        int my_node_idx = *node_idx;
        *node_idx += 1;
        node_lvl[my_node_idx] = my_level;
        n_children[my_node_idx] = 0;
        pt_cluster[2 * my_node_idx] = x0idx;
        pt_cluster[2 * my_node_idx + 1] = x0idx + npt - 1;
        node_npt[my_node_idx] = npt;
        memcpy(enbox + my_node_idx * 2 * pt_dim, curr_enbox, sizeof(VT) * 2 * pt_dim);
        return my_node_idx;
    }

    // 3. Assign each point to a child node
    int *clu_idx = worki;
    int clu_cnt[OCTREE_MAX_CHILDREN];
    memset(clu_cnt, 0, sizeof(int) * max_children);
    memset(clu_idx, 0, sizeof(int) * npt);
    int pow2k = 1;
    for (int k = 0; k < pt_dim; k++)
    {
        VT *x_k = x + k * ldx;
        for (int i = 0; i < npt; i++)
            if (x_k[i] >= curr_center[k]) clu_idx[i] += pow2k;
        pow2k *= 2;
    }
    for (int i = 0; i < npt; i++) clu_cnt[clu_idx[i]]++;

    // 4. Compute the enbox of each child node
    VT clu_enbox[OCTREE_MAX_CHILDREN * 2 * OCTREE_MAX_DIM];
    pow2k = 1;
    for (int k = 0; k < pt_dim; k++)
    {
        for (int i = 0; i < max_children; i += pow2k * 2)
        {
            for (int j = i; j < i + 2 * pow2k; j++)
            {
                VT *clu_enbox_j = &clu_enbox[j * 2 * pt_dim];
                clu_enbox_j[k] = (j < i + pow2k) ? curr_enbox[k] : curr_center[k];
                clu_enbox_j[pt_dim + k] = curr_enbox_size[k] * 0.5;
            }
        }
        pow2k *= 2;
    }
    for (int j = 0; j < max_children; j++)
    {
        VT *clu_enbox_j = &clu_enbox[j * 2 * pt_dim];
        extend_enbox(pt_dim, clu_enbox_j);
    }

    // 5. Shuffle the points
    VT *x_tmp = workv;
    int *pidx_tmp = worki + npt;
    int *perm_idx = worki + npt * 2;
    int clu_displs[OCTREE_MAX_CHILDREN];
    clu_displs[0] = 0;
    for (int i = 1; i < max_children; i++) clu_displs[i] = clu_displs[i - 1] + clu_cnt[i - 1];
    for (int i = 0; i < npt; i++)
    {
        int idx = clu_displs[clu_idx[i]];
        clu_displs[clu_idx[i]]++;
        perm_idx[idx] = i;
        pidx_tmp[idx] = pidx[i];
    }
    for (int j = 0; j < pt_dim; j++)
    {
        VT *src = x + j * ldx;
        VT *dst = x_tmp + j * ldx;
        for (int i = 0; i < npt; i++) dst[i] = src[perm_idx[i]];
    }
    clu_displs[0] = 0;
    for (int i = 1; i < max_children; i++) clu_displs[i] = clu_displs[i - 1] + clu_cnt[i - 1];
    for (int j = 0; j < pt_dim; j++)
    {
        VT *src = x_tmp + j * ldx;
        VT *dst = x + j * ldx;
        memcpy(dst, src, sizeof(VT) * npt);
    }
    memcpy(pidx, pidx_tmp, sizeof(int) * npt);

    // 5. Recursively partition the points in each cluster
    int children_cnt = 0;
    int children_nidx[OCTREE_MAX_CHILDREN];
    for (int i = 0; i < max_children; i++)
    {
        if (clu_cnt[i] == 0) continue;
        int disp = clu_displs[i];
        VT *clu_enbox_i = &clu_enbox[i * 2 * pt_dim];
        children_nidx[children_cnt] = octree_recursive_partition<VT>(
            clu_cnt[i], pt_dim, leaf_nmax, leaf_emax, 
            clu_enbox_i, x + disp, ldx, x0idx + disp, pidx + disp, 
            my_level + 1, node_lvl, parent, children, n_children, 
            enbox, pt_cluster, node_npt, node_idx, workv, worki
        );
        children_cnt++;
    }

    // 6. Add current node to the tree
    int my_node_idx = *node_idx;
    *node_idx += 1;
    node_lvl[my_node_idx] = my_level;
    n_children[my_node_idx] = children_cnt;
    pt_cluster[2 * my_node_idx] = x0idx;
    pt_cluster[2 * my_node_idx + 1] = x0idx + npt - 1;
    node_npt[my_node_idx] = npt;
    for (int i = 0; i < children_cnt; i++)
    {
        children[my_node_idx * max_children + i] = children_nidx[i];
        parent[children_nidx[i]] = my_node_idx;
    }
    memcpy(enbox + my_node_idx * 2 * pt_dim, curr_enbox, sizeof(VT) * 2 * pt_dim);

    return my_node_idx;
}

template<typename VT>
static void octree_build(
    const int npt, const int pt_dim, const int val_type, const VT *coord, 
    const int leaf_nmax, const VT leaf_emax, octree_p *octree_
)
{
    ASSERT_PRINTF(
        pt_dim <= OCTREE_MAX_DIM && pt_dim >= 1,
        "Expected 1 <= pt_dim <= %d, got pt_dim = %d\n", OCTREE_MAX_DIM, pt_dim
    );

    octree_p octree = (octree_p) malloc(sizeof(octree_s));
    *octree_ = octree;
    octree->npt = npt;
    octree->pt_dim = pt_dim;
    int max_children = 1 << pt_dim;

    // 1. Find a cubic enclosing box of the point set, so the enclosing box of 
    //    all nodes are also cubic (required for proxy point generation)
    VT root_enbox[2 * OCTREE_MAX_DIM];
    calc_min_enbox<VT>(npt, pt_dim, coord, npt, &root_enbox[0]);
    VT max_enbox_size = *std::max_element(&root_enbox[0] + pt_dim, &root_enbox[0] + 2 * pt_dim);
    max_enbox_size *= (1.0 + 2e-5);
    for (int i = 0; i < pt_dim; i++)
    {
        VT center_i = root_enbox[i] + root_enbox[pt_dim + i] * 0.5;
        root_enbox[i] = center_i - max_enbox_size * 0.5;
        root_enbox[pt_dim + i] = max_enbox_size;
    }

    // 2. Recursively partition the points using a given enclosing box
    //    to ensure that for all nodes at the same level, their enclosing boxes
    //    have the same size (required for proxy point generation)
    int *worki0 = (int *) malloc(sizeof(int) * npt * (max_children + 9));
    VT  *workv0 = (VT  *) malloc(sizeof(VT)  * npt * pt_dim * 4);
    assert(worki0 != NULL && workv0 != NULL);
    int *pidx       = worki0;
    int *node_lvl   = pidx       + npt;
    int *parent     = node_lvl   + npt;
    int *children   = parent     + npt;
    int *n_children = children   + npt * max_children;
    int *pt_cluster = n_children + npt;
    int *node_npt   = pt_cluster + npt;
    int *worki      = node_npt   + npt;  // npt * 3
    VT  *x1         = workv0;
    VT  *enbox      = x1    + npt * pt_dim;
    VT  *workv      = enbox + npt * pt_dim * 2;  // npt * pt_dim
    int node_idx = 0;
    for (int i = 0; i < npt; i++) pidx[i] = i;
    memcpy(x1, coord, sizeof(VT) * pt_dim * npt);
    int root_idx = octree_recursive_partition<VT>(
        npt, pt_dim, leaf_nmax, leaf_emax,
        &root_enbox[0], x1, npt, 0, pidx, 
        0, node_lvl, parent, children, n_children, 
        enbox, pt_cluster, node_npt, &node_idx, workv, worki
    );
    parent[root_idx] = -1;  // Root node has no parent

    // 2. Copy partitioning tree data
    int n_node = node_idx;
    int max_level = *std::max_element(node_lvl, node_lvl + n_node);
    int n_level = max_level + 1;
    octree->n_node   = n_node;
    octree->n_level  = n_level;
    octree->val_type = val_type;
    octree->parent     = (int *) malloc(sizeof(int) * n_node);
    octree->children   = (int *) malloc(sizeof(int) * n_node * max_children);
    octree->n_children = (int *) malloc(sizeof(int) * n_node);
    octree->node_lvl   = (int *) malloc(sizeof(int) * n_node);
    octree->node_npt   = (int *) malloc(sizeof(int) * n_node);
    octree->lvl_nnode  = (int *) malloc(sizeof(int) * n_level);
    octree->lvl_nodes  = (int *) malloc(sizeof(int) * n_node);
    octree->ln_displs  = (int *) malloc(sizeof(int) * (n_level + 1));
    octree->pt_cluster = (int *) malloc(sizeof(int) * n_node * 2);
    octree->fwd_perm   = (int *) malloc(sizeof(int) * npt);
    octree->bwd_perm   = (int *) malloc(sizeof(int) * npt);
    octree->px         = (VT  *) malloc(sizeof(VT)  * npt * pt_dim);
    octree->enbox      = (VT  *) malloc(sizeof(VT)  * n_node * pt_dim * 2);
    memcpy(octree->parent,     parent,     sizeof(int) * n_node);
    memcpy(octree->children,   children,   sizeof(int) * n_node * max_children);
    memcpy(octree->n_children, n_children, sizeof(int) * n_node);
    memcpy(octree->node_lvl,   node_lvl,   sizeof(int) * n_node);
    memcpy(octree->node_npt,   node_npt,   sizeof(int) * n_node);
    memcpy(octree->pt_cluster, pt_cluster, sizeof(int) * n_node * 2);
    memcpy(octree->px,         x1,         sizeof(VT)  * npt * pt_dim);
    memcpy(octree->enbox,      enbox,      sizeof(VT)  * n_node * pt_dim * 2);
    memcpy(octree->bwd_perm,   pidx,       sizeof(int) * npt);
    for (int i = 0; i < npt; i++) octree->fwd_perm[pidx[i]] = i;
    int *lvl_nnode = octree->lvl_nnode;
    int *lvl_nodes = octree->lvl_nodes;
    int *ln_displs = octree->ln_displs;
    memset(lvl_nnode, 0, sizeof(int) * n_level);
    for (int i = 0; i < n_node; i++) lvl_nnode[node_lvl[i]]++;
    ln_displs[0] = 0;
    for (int i = 1; i <= n_level; i++) 
        ln_displs[i] = ln_displs[i - 1] + lvl_nnode[i - 1];
    for (int i = 0; i < n_node; i++)
    {
        int idx = ln_displs[node_lvl[i]];
        ln_displs[node_lvl[i]]++;
        lvl_nodes[idx] = i;
    }
    ln_displs[0] = 0;
    for (int i = 1; i <= n_level; i++) 
        ln_displs[i] = ln_displs[i - 1] + lvl_nnode[i - 1];

    free(worki0);
    free(workv0);
}

void octree_build(
    const int npt, const int pt_dim, const int val_type, const void *coord, 
    const int leaf_nmax, const void *leaf_emax, octree_p *octree
)
{
    if (val_type == VAL_TYPE_DOUBLE)
    {
        const double *leaf_emax_ = (const double *) leaf_emax;
        octree_build<double>(npt, pt_dim, val_type, (const double *) coord, leaf_nmax, leaf_emax_[0], octree);
    }
    if (val_type == VAL_TYPE_FLOAT)
    {
        const float *leaf_emax_ = (const float *) leaf_emax;
        octree_build<float> (npt, pt_dim, val_type, (const float *)  coord, leaf_nmax, leaf_emax_[0], octree);
    }
}

// Free an octree struct
void octree_free(octree_p *octree)
{
    octree_p octree_ = *octree;
    if (octree_ == NULL) return;
    free(octree_->parent);
    free(octree_->children);
    free(octree_->n_children);
    free(octree_->node_lvl);
    free(octree_->node_npt);
    free(octree_->lvl_nnode);
    free(octree_->lvl_nodes);
    free(octree_->ln_displs);
    free(octree_->pt_cluster);
    free(octree_->fwd_perm);
    free(octree_->bwd_perm);
    free(octree_->px);
    free(octree_->enbox);
    free(octree_);
    *octree = NULL;
}
