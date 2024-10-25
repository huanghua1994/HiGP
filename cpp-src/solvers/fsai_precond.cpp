#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <type_traits>
#include <omp.h>

#include "fsai_precond.h"
#include "../common.h"
#include "../cpu_linalg.hpp"
#include "../utils.h"
#include "../dense_kernel_matrix.h"
#include "csr_mat.h"

template<typename VT>
static void fsai_exact_knn(
    const int fsai_npt, const int n, const int pt_dim, 
    const VT *coord, const int ldc, int *nn_idx, int *nn_cnt
)
{
    int n_thread = omp_get_max_threads();
    memset(nn_cnt, 0, sizeof(int) * n);
    int *idx_buf   = (int *) malloc(sizeof(int) * n_thread * n);
    VT  *dist2_buf = (VT *)  malloc(sizeof(VT)  * n_thread * n);
    ASSERT_PRINTF(idx_buf != NULL && dist2_buf != NULL, "Failed to allocate work arrays for %s\n", __FUNCTION__);
    
    VT param = (VT) pt_dim;
    int val_type  = std::is_same<VT, double>::value ? VAL_TYPE_DOUBLE : VAL_TYPE_FLOAT;

    #pragma omp parallel if(n_thread > 1) num_threads(n_thread)
    {
        int tid = omp_get_thread_num();
        int *idx = idx_buf + tid * n;
        VT *dist2 = dist2_buf + tid * n;
        // Do the large chunk of work first
        #pragma omp for schedule(dynamic, 256)
        for (int i = n - 1; i >= 0; i--)
        {
            int nn_cnt_i = 0;
            int *nn_idx_i = nn_idx + i * fsai_npt;
            if (i < fsai_npt)
            {
                nn_cnt_i = i + 1;
                for (int j = 0; j < nn_cnt_i; j++) nn_idx_i[j] = j;
            } else {
                nn_cnt_i = fsai_npt;
                for (int j = 0; j < i; j++) idx[j] = j;
                pdist2_krnl(
                    i, ldc, coord, 1, ldc, coord + i, 
                    (void *) &param, i, dist2, val_type
                );
                qpart_key_val<VT, int>(dist2, idx, 0, i - 1, fsai_npt);
                memcpy(nn_idx_i, idx, sizeof(int) * fsai_npt);
                nn_idx_i[nn_cnt_i - 1] = i;
            }
            nn_cnt[i] = nn_cnt_i;
        }
    }
    free(idx_buf);
    free(dist2_buf);
}

// Select k exact nearest neighbors for each point s.t. the indices of 
// neighbors are smaller than the index of the point
void fsai_exact_knn(
    const int val_type, const int fsai_npt, const int n, const int pt_dim, 
    const void *coord, const int ldc, int *nn_idx, int *nn_cnt
)
{
    if (val_type == VAL_TYPE_DOUBLE) fsai_exact_knn<double>(fsai_npt, n, pt_dim, (const double *) coord, ldc, nn_idx, nn_cnt);
    if (val_type == VAL_TYPE_FLOAT)  fsai_exact_knn<float> (fsai_npt, n, pt_dim, (const float *)  coord, ldc, nn_idx, nn_cnt);
}

// Check if segments [s1, s1+l1] and [s2, s2+l2] are neighbors
template<typename VT>
static inline int is_neighbor_segments(const VT s1, const VT l1, const VT s2, const VT l2)
{
    int is_neighbor = 0;
    VT diff_s = std::abs(s2 - s1);
    if (s1 <= s2) is_neighbor = (diff_s / l1 < 1.00001);
    else is_neighbor = (diff_s / l2 < 1.00001);
    return is_neighbor;
}

template<typename VT>
static void fsai_octree_fast_knn(
    const int fsai_npt, const int n, const int pt_dim, const VT *coord, const int ldc, 
    const int *coord0_idx, octree_p octree, int *nn_idx, int *nn_cnt
)
{
    int n_thread = omp_get_max_threads();
    memset(nn_cnt, 0, sizeof(int) * n);

    VT param = (VT) pt_dim;
    int val_type  = std::is_same<VT, double>::value ? VAL_TYPE_DOUBLE : VAL_TYPE_FLOAT;

    // 1. Find the highest level of octree leaf nodes, set it as search level,
    //    and find the maximum number of points in a node at search level
    int search_lvl = octree->n_level - 1;
    int *n_children = octree->n_children;
    for (int i = 0; i < octree->n_node; i++)
    {
        if (n_children[i] > 0) continue;
        int node_lvl = octree->node_lvl[i];
        if (node_lvl < search_lvl) search_lvl = node_lvl;
    }
    int max_sl_node_npt = 0;
    int search_lvl_nnode = octree->lvl_nnode[search_lvl];
    int *search_lvl_nodes = octree->lvl_nodes + octree->ln_displs[search_lvl];
    int *pt_cluster = octree->pt_cluster;  // Notice: each pair in pt_cluster is [start, end] not [start, end)
    for (int i = 0; i < search_lvl_nnode; i++)
    {
        int node = search_lvl_nodes[i];
        int node_npt = pt_cluster[2 * node + 1] - pt_cluster[2 * node] + 1;
        if (node_npt > max_sl_node_npt) max_sl_node_npt = node_npt;
    }

    // 2. Create a mapping for points in the H2 tree to the coord array
    //    OIPS in the comments == "original input point set"
    int n0 = octree->npt;  // Number of points in the OIPS
    int *octree_cidx   = octree->bwd_perm;                   // For each point in the octree, its index in the OIPS
    int *idx0_to_idx   = (int *) malloc(sizeof(int) * n0);   // For each point in the OIPS, its index in coord
    int *octree_to_idx = (int *) malloc(sizeof(int) * n0);   // For each point in the octree, its index in coord
    ASSERT_PRINTF(
        idx0_to_idx != NULL && octree_to_idx != NULL, 
        "Failed to allocate work arrays for %s\n", __FUNCTION__
    );
    for (int i = 0; i < n0; i++) idx0_to_idx[i] = -1;
    for (int i = 0; i < n;  i++) idx0_to_idx[coord0_idx[i]] = i;
    for (int i = 0; i < n0; i++) octree_to_idx[i] = idx0_to_idx[octree_cidx[i]];

    // 3. Find all neighbor nodes (including self) for each node at search level.
    //    Each node has at most 3^pt_dim neighbors.
    int max_neighbor = 1;
    for (int i = 0; i < octree->pt_dim; i++) max_neighbor *= 3;
    int *sl_node_neighbors    = (int *) malloc(sizeof(int) * search_lvl_nnode * max_neighbor);
    int *sl_node_neighbor_cnt = (int *) malloc(sizeof(int) * search_lvl_nnode);
    ASSERT_PRINTF(
        sl_node_neighbors != NULL && sl_node_neighbor_cnt != NULL, 
        "Failed to allocate work arrays for %s\n", __FUNCTION__
    );
    VT *enbox = (VT *) octree->enbox;
    memset(sl_node_neighbor_cnt, 0, sizeof(int) * search_lvl_nnode);
    for (int i = 0; i < search_lvl_nnode; i++)
    {
        int node_i = search_lvl_nodes[i];
        sl_node_neighbors[i * max_neighbor] = node_i;
        sl_node_neighbor_cnt[i] = 1;
    }
    for (int i = 0; i < search_lvl_nnode; i++)
    {
        int node_i = search_lvl_nodes[i];
        VT *enbox_i = enbox + (2 * pt_dim) * node_i;
        for (int j = i + 1; j < search_lvl_nnode; j++)
        {
            int node_j = search_lvl_nodes[j];
            VT *enbox_j = enbox + (2 * pt_dim) * node_j;
            int is_neighbor = 1;
            for (int d = 0; d < pt_dim; d++)
            {
                VT sid = enbox_i[d];
                VT lid = enbox_i[d + pt_dim];
                VT sjd = enbox_j[d];
                VT ljd = enbox_j[d + pt_dim];
                int is_neighbor_d = is_neighbor_segments(sid, lid, sjd, ljd);
                is_neighbor = is_neighbor && is_neighbor_d;
            }
            if (is_neighbor)
            {
                int cnt_i = sl_node_neighbor_cnt[i];
                int cnt_j = sl_node_neighbor_cnt[j];
                ASSERT_PRINTF(cnt_i < max_neighbor, "Node %d has more than %d neighbors\n", node_i, max_neighbor);
                ASSERT_PRINTF(cnt_j < max_neighbor, "Node %d has more than %d neighbors\n", node_j, max_neighbor);
                sl_node_neighbors[i * max_neighbor + cnt_i] = node_j;
                sl_node_neighbors[j * max_neighbor + cnt_j] = node_i;
                sl_node_neighbor_cnt[i]++;
                sl_node_neighbor_cnt[j]++;
            }
        }  // End of j loop
    }  // End of i loop

    // 4. Find the nearest neighbors from sl_node_neighbors for each point.
    //    The nearest neighbor candidates of each point should <= max_neighbor * max_sl_node_npt
    int  max_nn_points = max_neighbor * max_sl_node_npt;
    int  *idx_buf      = (int *)  malloc(sizeof(int)  * n_thread * max_nn_points * 2);
    char *flag_buf     = (char *) malloc(sizeof(char) * n_thread * n);
    VT   *dist2_buf    = (VT *)   malloc(sizeof(VT)   * n_thread * max_nn_points);
    VT   *nn_coord_buf = (VT *)   malloc(sizeof(VT)   * n_thread * max_nn_points * pt_dim);
    ASSERT_PRINTF(
        idx_buf != NULL && dist2_buf != NULL && nn_coord_buf != NULL,
        "Failed to allocate work arrays for %s\n", __FUNCTION__
    );
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        VT   *dist2    = dist2_buf + tid * max_nn_points;
        VT   *nn_coord = nn_coord_buf + tid * pt_dim * max_nn_points;
        int  *nn_idx0  = idx_buf + tid * max_nn_points * 2;
        int  *nn_idx1  = nn_idx0 + max_nn_points;
        char *flag     = flag_buf + tid * n;
        #pragma omp for schedule(dynamic)
        for (int i = 0; i < search_lvl_nnode; i++)
        {
            int node_i = search_lvl_nodes[i];
            int nn_cnt0 = 0;
            // (1) Put all points in node_s's neighbor nodes into a large candidate set
            for (int j = 0; j < sl_node_neighbor_cnt[i]; j++)
            {
                int node_j = sl_node_neighbors[i * max_neighbor + j];
                int node_j_pt_s = pt_cluster[2 * node_j];
                int node_j_pt_e = pt_cluster[2 * node_j + 1];
                for (int k = node_j_pt_s; k <= node_j_pt_e; k++)
                {
                    int idx_k = octree_to_idx[k];   // The k-th point in H2 tree, its index in coord
                    if (idx_k == -1) continue;
                    nn_idx0[nn_cnt0++] = idx_k;
                }
            }
            // (2) For each point in this node, find its KNN in the refined candidate set
            int node_i_pt_s = pt_cluster[2 * node_i];
            int node_i_pt_e = pt_cluster[2 * node_i + 1];
            for (int pt_j = node_i_pt_s; pt_j <= node_i_pt_e; pt_j++)
            {
                int idx_j = octree_to_idx[pt_j];   // The pt_j-th point in H2 tree, its index in coord
                if (idx_j == -1) continue;
                // All point indices in nn_idx0 are in coord, no need to translate index again
                // FSAI requires the NN points have indices < current point
                int nn_cnt1 = 0;
                for (int k = 0; k < nn_cnt0; k++)
                    if (nn_idx0[k] < idx_j) nn_idx1[nn_cnt1++] = nn_idx0[k];
                // If the number of NN candidates < fsai_npt, expand the candidate set with the first
                // (max_nn_points - nn_cnt1) points (this is the simplest way, but may miss some exact NNs)
                if ((nn_cnt1 < fsai_npt) && (idx_j >= fsai_npt))
                {
                    memset(flag, 0, sizeof(char) * n);
                    for (int k = 0; k < nn_cnt1; k++) flag[nn_idx1[k]] = 1;
                    for (int k = 0; k < idx_j; k++)
                    {
                        if (flag[k] == 1) continue;
                        nn_idx1[nn_cnt1++] = k;
                        if (nn_cnt1 == max_nn_points - 1) break;
                    }
                }
                int *nn_idx_j = nn_idx + idx_j * fsai_npt;
                if (idx_j < fsai_npt)
                {
                    nn_cnt[idx_j] = idx_j + 1;
                    for (int j = 0; j <= idx_j; j++) nn_idx_j[j] = j;
                } else {
                    int nn_cnt_j = (fsai_npt < nn_cnt1) ? fsai_npt : nn_cnt1;
                    nn_cnt[idx_j] = nn_cnt_j;
                    // gather_matrix_cols() works on row-major matrix
                    gather_matrix_cols(sizeof(VT), pt_dim, nn_cnt1, nn_idx1, coord, ldc, nn_coord, nn_cnt1);
                    pdist2_krnl(
                        nn_cnt1, nn_cnt1, nn_coord, 1, ldc, coord + idx_j, 
                        (void *) &param, nn_cnt1, dist2, val_type
                    );
                    qpart_key_val<VT, int>(dist2, nn_idx1, 0, nn_cnt1 - 1, fsai_npt);
                    memcpy(nn_idx_j, nn_idx1, sizeof(int) * nn_cnt_j);
                    nn_idx_j[nn_cnt_j - 1] = idx_j;
                }
            }  // End of pt_j loop
        }  // End of i loop
    }  // End of "#pragma omp parallel"

    // Sanity check
    int invalid_cnt = 0;
    for (int i = 0; i < n; i++)
        if (nn_cnt[i] < 1 || nn_cnt[i] > fsai_npt) invalid_cnt++;
    ASSERT_PRINTF(invalid_cnt == 0, "%d points have invalid NN count\n", invalid_cnt);

    free(sl_node_neighbors);
    free(sl_node_neighbor_cnt);
    free(idx0_to_idx);
    free(octree_to_idx);
    free(dist2_buf);
    free(flag_buf);
    free(nn_coord_buf);
    free(idx_buf);
}

// Select k (approximate) nearest neighbors for each point s.t. the 
// indices of neighbors are smaller than the index of the point
void fsai_octree_fast_knn(
    const int val_type, const int fsai_npt, const int n, const int pt_dim, 
    const void *coord, const int ldc, const int *coord0_idx, octree_p octree, 
    int *nn_idx, int *nn_cnt
)
{
    if (val_type == VAL_TYPE_DOUBLE)
    {
        fsai_octree_fast_knn<double>(
            fsai_npt, n, pt_dim, (const double *) coord, ldc, 
            coord0_idx, octree, nn_idx, nn_cnt
        );
    }
    if (val_type == VAL_TYPE_FLOAT)
    {
        fsai_octree_fast_knn<float>(
            fsai_npt, n, pt_dim, (const float *)  coord, ldc, 
            coord0_idx, octree, nn_idx, nn_cnt
        );
    }
}

template<typename VT>
static void fsai_precond_build(
    const int val_type, const int krnl_id, const VT *param, const VT *dnoise, 
    const int npt, const int pt_dim, const VT *coord, const int ldc, 
    const int fsai_npt, const int *nn_idx, const int *nn_displs, 
    const int n1, const VT *P, const int need_grad, 
    const VT *GdK12, const VT *GdV12, fsai_precond_p *fp
)
{
    fsai_precond_p fp_ = (fsai_precond_p) malloc(sizeof(fsai_precond_s));
    memset(fp_, 0, sizeof(fsai_precond_s));
    fp_->n = npt;
    fp_->val_type = val_type;

    int n_grad = need_grad ? 3 : 0;
    size_t VT_bytes = sizeof(VT);

    // 1. Allocate work buffer
    int n_thread = omp_get_max_threads();
    int thread_bufsize = 0;
    thread_bufsize += fsai_npt * pt_dim;    // Xnn
    thread_bufsize += fsai_npt * fsai_npt;  // tmpK
    thread_bufsize += n1 * fsai_npt;        // Pnn
    thread_bufsize += fsai_npt;             // tmpU
    thread_bufsize += fsai_npt;             // nn_dn
    int *row    = (int *) malloc(sizeof(int) * npt * fsai_npt);
    int *col    = (int *) malloc(sizeof(int) * npt * fsai_npt);
    int *G_idx  = (int *) malloc(sizeof(int) * npt * fsai_npt);
    int *GT_idx = (int *) malloc(sizeof(int) * npt * fsai_npt);
    VT  *val    = (VT *)  malloc(sizeof(VT)  * npt * fsai_npt);
    VT  *dGvals = NULL;
    if (need_grad)
    {
        thread_bufsize += n_grad * fsai_npt * fsai_npt;  // dKdl, dKdf, dKds
        thread_bufsize += n1 * fsai_npt;                 // GdK12ji
        thread_bufsize += n1 * fsai_npt;                 // GdV12ji
        thread_bufsize += fsai_npt * fsai_npt;           // tmpK1
        thread_bufsize += fsai_npt * fsai_npt;           // tmpK2
        thread_bufsize += fsai_npt;                      // idKe
        dGvals = (VT *) malloc(sizeof(VT) * npt * fsai_npt * n_grad);
        ASSERT_PRINTF(dGvals != NULL, "Failed to allocate work buffers for %s\n", __FUNCTION__);
    }
    VT *matbuf = (VT *) malloc(sizeof(VT) * n_thread * thread_bufsize);
    ASSERT_PRINTF(
        matbuf != NULL && row != NULL && col != NULL && 
        val != NULL && G_idx != NULL && GT_idx != NULL,
        "Failed to allocate work buffers for %s\n", __FUNCTION__
    );

    // 2. Build the FSAI COO matrix
    int i_one = 1;
    VT v_neg_one = -1.0, v_one = 1.0, v_zero = 0.0;
    #pragma omp parallel if (n_thread > 1) num_threads(n_thread)
    {
        int tid = omp_get_thread_num();
        VT *t_matbuf = matbuf + tid * thread_bufsize;
        VT *nn_coord = t_matbuf;
        VT *tmpK     = nn_coord + fsai_npt * pt_dim;
        VT *Pnn      = tmpK     + fsai_npt * fsai_npt;
        VT *tmpU     = Pnn      + n1 * fsai_npt;
        VT *nn_dn    = tmpU     + fsai_npt;
        VT *dtmpK = NULL, *dKdl = NULL, *dKdf = NULL, *dKds = NULL;
        VT *GdK12ji = NULL, *GdV12ji = NULL, *tmpK1 = NULL, *tmpK2 = NULL, *idKe = NULL;
        if (need_grad)
        {
            dtmpK   = nn_dn   + fsai_npt;
            dKdl    = dtmpK;
            dKdf    = dKdl    + fsai_npt * fsai_npt;
            dKds    = dKdf    + fsai_npt * fsai_npt;
            GdK12ji = dtmpK   + n_grad * fsai_npt * fsai_npt;
            GdV12ji = GdK12ji + n1 * fsai_npt;
            tmpK1   = GdV12ji + n1 * fsai_npt;
            tmpK2   = tmpK1   + fsai_npt * fsai_npt;
            idKe    = tmpK2   + fsai_npt * fsai_npt;
        }

        // In the first fsai_npt rows, each row has less than fsai_npt nonzeros;
        // after that, each row has exactly fsai_npt nonzeros. Using a static
        // partitioning scheme should be good enough.
        #pragma omp for schedule(static)
        for (int i = 0; i < npt; i++)
        {
            int nn_cnt_i  = nn_displs[i + 1] - nn_displs[i];
            int j_start   = nn_displs[i];
            const int *nn_idx_i = nn_idx + i * fsai_npt;

            // row(idx + (1 : nn_cnt_i)) = i;
            // col(idx + (1 : nn_cnt_i)) = nn;
            // Xnn = X(nn, :);
            for (int j = j_start; j < j_start + nn_cnt_i; j++)
            {
                row[j]    = i;
                col[j]    = nn_idx_i[j - j_start];
                G_idx[j]  = j;
                GT_idx[j] = j;
            }
            // gather_matrix_cols works on row-major matrices
            gather_matrix_cols(VT_bytes, pt_dim, nn_cnt_i, nn_idx_i, (void *) coord,  ldc, (void *) nn_coord, nn_cnt_i);
            gather_matrix_cols(VT_bytes, 1,      nn_cnt_i, nn_idx_i, (void *) dnoise, 1,   (void *) nn_dn,    nn_cnt_i);

            // tmpK = f^2 * (kernel(Xnn, Xnn) + s * I);
            dense_krnl_mat_p dK_Xnn = NULL;
            dense_krnl_mat_init(
                nn_cnt_i, nn_cnt_i, nn_coord, nn_cnt_i, nn_cnt_i, nn_coord,
                (const void*) param, nn_dn, krnl_id, val_type, &dK_Xnn
            );
            dense_krnl_mat_grad_eval(dK_Xnn, (void *) tmpK, (void *) dKdl, (void *) dKdf, (void *) dKds);
            dense_krnl_mat_free(&dK_Xnn);

            // tmpK = tmpK - P(: nn)' * P(:, nn);
            if (n1 > 0)
            {
                // gather_matrix_rows works on row-major matrices
                gather_matrix_rows(VT_bytes, nn_cnt_i, n1, nn_idx_i, (const void *) P, n1, (void *) Pnn, n1);
                xsyrk_(lower, trans, &nn_cnt_i, &n1, &v_neg_one, Pnn, &n1, &v_one, tmpK, &nn_cnt_i);
            }  // End of "if (n1 > 0)"

            // tmpU = [zeros(nn_cnt_i-1, 1); 1];
            // tmpY = tmpK \ tmpU;
            int info;
            VT *tmpY = tmpU;
            memset(tmpU, 0, sizeof(VT) * nn_cnt_i);
            tmpU[nn_cnt_i - 1] = 1.0;
            xposv_(lower, &nn_cnt_i, &i_one, tmpK, &nn_cnt_i, tmpU, &nn_cnt_i, &info);
            ASSERT_PRINTF(info == 0, "tid %d i = %d LAPACK xPOSV return %d\n", tid, i, info);

            // d2 = sqrt(tmpY(nn_cnt_i));
            // val(idx + (1 : nn_cnt_i)) = tmpY / d2;
            VT inv_d2 = 1.0 / std::sqrt(tmpY[nn_cnt_i - 1]);
            for (int j = 0; j < nn_cnt_i; j++) val[j + j_start] = tmpY[j] * inv_d2;

            for (int j = 0; j < n_grad; j++)
            {
                VT *dKj = dtmpK + j * fsai_npt * fsai_npt;
                if (n1 > 0)
                {
                    // GdK12ji = GdK12{j}(:, nn);
                    // GdV12ji = GdV12{j}(:, nn);
                    const VT *GdK12j = GdK12 + j * n1 * npt;
                    const VT *GdV12j = GdV12 + j * n1 * npt;
                    // gather_matrix_rows works on row-major matrices
                    gather_matrix_rows(VT_bytes, nn_cnt_i, n1, nn_idx_i, (void *) GdK12j, n1, (void *) GdK12ji, n1);
                    gather_matrix_rows(VT_bytes, nn_cnt_i, n1, nn_idx_i, (void *) GdV12j, n1, (void *) GdV12ji, n1);
                    // tmpK1 = Pnn' * GdK12ji;
                    // tmpK2 = Pnn' * GdV12ji;
                    xgemm_(
                        trans, notrans, &nn_cnt_i, &nn_cnt_i, &n1,
                        &v_one, Pnn, &n1, GdK12ji, &n1,
                        &v_zero, tmpK1, &nn_cnt_i
                    );
                    xgemm_(
                        trans, notrans, &nn_cnt_i, &nn_cnt_i, &n1,
                        &v_one, Pnn, &n1, GdV12ji, &n1,
                        &v_zero, tmpK2, &nn_cnt_i
                    );
                    // dKj = dKj - tmpK1 - tmpK1' + tmpK2;
                    for (int ii = 0; ii < nn_cnt_i; ii++)
                    {
                        for (int jj = 0; jj < nn_cnt_i; jj++)
                        {
                            int idx0 = ii * nn_cnt_i + jj;
                            int idx1 = jj * nn_cnt_i + ii;
                            dKj[idx0] = dKj[idx0] - tmpK1[idx0] - tmpK1[idx1] + tmpK2[idx0];
                        }
                    }
                }  // End of "if (n1 > 0)"
                // idKe = -tmpK \ (dKj * tmpY);
                // Note: tmpK now stores its lower Cholesky factor
                xgemv_(
                    notrans, &nn_cnt_i, &nn_cnt_i, &v_one, dKj, &nn_cnt_i,
                    tmpY, &i_one, &v_zero, idKe, &i_one
                );
                xpotrs_(lower, &nn_cnt_i, &i_one, tmpK, &nn_cnt_i, idKe, &nn_cnt_i, &info);
                ASSERT_PRINTF(info == 0, "tid %d i = %d xPOTRS return %d\n", tid, i, info);
                #pragma omp simd
                for (int ii = 0; ii < nn_cnt_i; ii++) idKe[ii] = -idKe[ii];
                // d3 = idKe(num_nn);
                // d4 = d3 / (2 * d2^3);
                VT d3 = idKe[nn_cnt_i - 1];
                VT d4 = d3 * 0.5 * inv_d2 * inv_d2 * inv_d2;
                // dG_vals(idx + (1 : nn_cnt_i), j) = idKe / d2 - d4 * tmpY;
                VT *dGvals_j = dGvals + j * npt * fsai_npt;
                #pragma omp simd
                for (int ii = 0; ii < nn_cnt_i; ii++) 
                    dGvals_j[ii + j_start] = idKe[ii] * inv_d2 - d4 * tmpY[ii];
            }  // End of j loop
        }  // End of i loop
    }  // End of "#pragma omp parallel"
    free(matbuf);

    // 3. Convert the FSAI COO matrix to CSR format
    csr_mat_p G_pat = NULL, GT_pat = NULL;
    coo_to_csr(VAL_TYPE_INT, npt, npt, nn_displs[npt], row, col, (const void *) G_idx,  &G_pat);
    coo_to_csr(VAL_TYPE_INT, npt, npt, nn_displs[npt], col, row, (const void *) GT_idx, &GT_pat);
    csr_trsm_build_tree(lower, G_pat);
    csr_trsm_build_tree(upper, GT_pat);
    csr_mat_p G = NULL, GT = NULL;
    csr_mat_p *dG = NULL, *dGT = NULL;
    dG  = (csr_mat_p *) malloc(sizeof(csr_mat_p) * n_grad);
    dGT = (csr_mat_p *) malloc(sizeof(csr_mat_p) * n_grad);
    csr_build_from_pattern(val_type, G_pat,  (const void *) val, &G);
    csr_build_from_pattern(val_type, GT_pat, (const void *) val, &GT);
    for (int i = 0; i < n_grad; i++)
    {
        VT *dGvals_i = dGvals + i * npt * fsai_npt;
        csr_build_from_pattern(val_type, G_pat,  (const void *) dGvals_i, &dG[i]);
        csr_build_from_pattern(val_type, GT_pat, (const void *) dGvals_i, &dGT[i]);
    }
    free(row);
    free(col);
    free(G_idx);
    free(GT_idx);
    free(val);
    free(dGvals);

    csr_mat_free(&G_pat);
    csr_mat_free(&GT_pat);
    fp_->G   = G;
    fp_->GT  = GT;
    fp_->dG  = dG;
    fp_->dGT = dGT;
    *fp = fp_;
}

// Build a Factorized Sparse Approximate Inverse (FSAI) preconditioner for a kernel 
// matrix f^2 * (K(X, X, l) + s * I) + P^T * P, where P is a low rank matrix
void fsai_precond_build(
    const int val_type, const int krnl_id, const void *param, const void *dnoise, 
    const int npt, const int pt_dim, const void *coord, const int ldc, 
    const int fsai_npt, const int *nn_idx, const int *nn_displs, 
    const int n1, const void *P, const int need_grad, 
    const void *GdK12, const void *GdV12, fsai_precond_p *fp
)
{
    if (val_type == VAL_TYPE_DOUBLE)
    {
        fsai_precond_build<double>(
            val_type, krnl_id, (const double *) param, (const double *) dnoise, 
            npt, pt_dim, (const double *) coord, ldc, 
            fsai_npt, nn_idx, nn_displs, 
            n1, (const double *) P, need_grad, 
            (const double *) GdK12, (const double *) GdV12, fp
        );
    }
    if (val_type == VAL_TYPE_FLOAT)
    {
        fsai_precond_build<float>(
            val_type, krnl_id, (const float *)  param, (const float *)  dnoise, 
            npt, pt_dim, (const float *)  coord, ldc, 
            fsai_npt, nn_idx, nn_displs, 
            n1, (const float *)  P, need_grad, 
            (const float *)  GdK12, (const float *)  GdV12, fp
        );
    }
}

// Free an initialized fsai_precond struct
void fsai_precond_free(fsai_precond_p *fp)
{
    fsai_precond_p fp_ = *fp;
    if (fp_ == NULL) return;
    csr_mat_free(&fp_->G);
    csr_mat_free(&fp_->GT);
    if (fp_->dG != NULL)
    {
        for (int i = 0; i < 3; i++)
        {
            csr_mat_free(&fp_->dG[i]);
            csr_mat_free(&fp_->dGT[i]);
        }
        free(fp_->dG);
        free(fp_->dGT);
    }
    free(fp_);
    *fp = NULL;
}

// Apply the FSAI preconditioner to multiple column vectors
void fsai_precond_apply(const void *fp, const int n, const void *B, const int ldB, void *C, const int ldC)
{
    fsai_precond_p fp_ = (fsai_precond_p) fp;
    if (fp_ == NULL) return;
    size_t VT_bytes = (fp_->val_type == VAL_TYPE_DOUBLE) ? sizeof(double) : sizeof(float);
    void *T = malloc(VT_bytes * fp_->n * n);
    int ldT = fp_->n;
    csr_spmm(fp_->G,  n, B, ldB, T, ldT);
    csr_spmm(fp_->GT, n, T, ldT, C, ldC);
    free(T);
}
