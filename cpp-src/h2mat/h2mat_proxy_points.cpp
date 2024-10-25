#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include "omp.h"

#include "h2mat_utils.h"
#include "h2mat_proxy_points.h"
#include "id_ppqr.h"
#include "../common.h"
#include "../kernels/kernels.h"

// Check if |K(r) / K(0)| < reltol, assuming that the kernel function 
// is translationally invariant and decreases monotonically w.r.t. r
// Input parameters:
//   krnl      : Pointer to kernel matrix evaluation function
//   param     : Pointer to kernel matrix evaluation function parameter list
//   pt_dim    : Point dimension
//   val_type  : Data type of coordinates and kernel matrix, 0 for double, 1 for float
//   dist      : Distance to check
//   reltol    : Relative tolerance
template<typename VT>
static int krnl_too_small(
    krnl_func krnl, const void *param, const int pt_dim, 
    const int val_type, const VT dist, const VT reltol
)
{
    VT *work = (VT *) malloc(sizeof(VT) * 2 * (pt_dim + 1));
    VT *c0 = work;
    VT *c1 = c0 + pt_dim;
    VT *k0 = c1 + pt_dim;
    VT *k1 = k0 + 1;
    memset(c0, 0, sizeof(VT) * pt_dim);
    memset(c1, 0, sizeof(VT) * pt_dim);
    c1[0] = dist;
    krnl(1, 1, (void *) c0, 1, 1, (void *) c0, param, 1, (void *) k0, val_type);
    krnl(1, 1, (void *) c0, 1, 1, (void *) c1, param, 1, (void *) k1, val_type);
    int res = std::abs(k1[0] / k0[0]) < reltol;
    free(work);
    return res;
}

// Generate proxy points with two domains specified as 
// X = [-L1/2, L1/2]^pt_dim, Y = [-L3/2, L3/2]^pt_dim \ [-L2/2, L2/2]^pt_dim
// generated proxy points are in domain Y
// Input parameters:
//   krnl       : Pointer to kernel matrix evaluation function
//   param      : Pointer to kernel matrix evaluation function parameter list
//   pt_dim     : Point dimension
//   val_type   : Data type of coordinates and kernel matrix, 0 for double, 1 for float
//   L1, L2, L3 : Domain sizes
//   *reltol    : Proxy point selection relative error tolerance
//   X0_size    : Number of proxy points in domain X
//   Y0_lsize   : Number of proxy points in each layer of domain Y0
//   max_layer  : Maximum number of layers in domain Y0
// Output parameter:
//   pp : Pointer to a h2m_2dbuf struct that stores the calculated proxy points
template<typename VT>
static void h2m_gen_proxy_points(
    krnl_func krnl, const void *param, const int pt_dim, const int val_type, 
    const VT L1, const VT L2, const VT L3, const VT *reltol, 
    const int X0_size, const int Y0_lsize, const int max_layer, h2m_2dbuf_p pp
)
{
    // Fast path: if the distance between X and Y are too far away, 
    // no proxy point is needed for this level
    VT gap_width = 0.5 * (L2 - L1), reltol1 = 0;
    if (val_type == VAL_TYPE_DOUBLE) reltol1 = 1e-15;
    if (val_type == VAL_TYPE_FLOAT)  reltol1 = 1e-6;
    if (krnl_too_small(krnl, param, pt_dim, val_type, gap_width, reltol1))
    {
        h2m_2dbuf_resize(pp, sizeof(VT), 0, pt_dim);
        return;
    }

    int n_layer = std::round((L3 - L2) / L1);
    if (n_layer > max_layer) n_layer = max_layer;
    int Y0_size = Y0_lsize * n_layer;
    h2m_2dbuf_p X0, Xp, Y0, Yi, Yp = pp;
    h2m_2dbuf_p tmpA, tmpA1, QR_buff, ID_buff;
    h2m_2dbuf_init(&X0,      sizeof(VT),  X0_size, pt_dim);
    h2m_2dbuf_init(&Xp,      sizeof(VT),  X0_size, pt_dim);
    h2m_2dbuf_init(&Y0,      sizeof(VT),  Y0_size, pt_dim);
    h2m_2dbuf_init(&Yi,      sizeof(VT),  Y0_size, pt_dim);
    h2m_2dbuf_init(&tmpA,    sizeof(VT),  Y0_size, X0_size);
    h2m_2dbuf_init(&tmpA1,   sizeof(VT),  X0_size, X0_size);
    h2m_2dbuf_init(&QR_buff, sizeof(VT),  Y0_size, 2);
    h2m_2dbuf_init(&ID_buff, sizeof(int), 1, Y0_size * 4);

    // If srand48() is not called before here, the generated proxy points may be too 
    // few (I see this with ICC but not GCC). I will just do a dirty hack here.
    srand(1924);
    srand48(1112);

    // 1. Generate initial candidate points in X and Y.
    //    For Y0, we generate it layer by layer. Each layer has the same number
    //    of points but different volume. So an inner layer has a higher density.
    VT v_zero = 0.0;
    VT Y0_layer_width = (L3 - L2) / (VT) n_layer;
    h2m_rand_points_in_shell(
        val_type, X0_size, pt_dim, (const void *) &v_zero,
        (const void *) &L1, X0->data, X0->nrow
    );
    VT *Y0_ptr = (VT *) Y0->data;
    for (int i = 0; i < n_layer; i++)
    {
        VT layer_L0 = L2 + Y0_layer_width * (VT) i;
        VT layer_L1 = L2 + Y0_layer_width * (VT) (i + 1);
        VT *Y0_i = Y0_ptr + i * Y0_lsize;
        h2m_rand_points_in_shell(
            val_type, Y0_lsize, pt_dim, (const void *) &layer_L0,
            (const void *) &layer_L1, Y0_i, Y0_size
        );
    }

    // 2. Select skeletion points in X0 using sparse randomized projection
    // (2.1) Generate tmpA = kernel(Y0, X0)
    krnl_func_omp(
        Y0_size, Y0_size, Y0->data, X0_size, X0_size, X0->data,
        krnl, param, tmpA->nrow, tmpA->data, val_type, 0
    );
    // (2.2) Generate sparse CSR matrix S of size X0_size * Y0_size
    int max_row_nnz = 32;
    h2m_2dbuf_p S_idx = ID_buff, S_val = QR_buff;
    h2m_sub_gaussian_csr(val_type, X0_size, Y0_size, max_row_nnz, S_idx, S_val);
    // (2.3) Compute tmpA1 = S * tmpA
    h2m_2dbuf_resize(tmpA1, sizeof(VT), X0_size, X0_size);
    int *row_ptr = S_idx->data_i;
    int *col_idx = row_ptr + (X0_size + 1);
    VT  *val     = (VT *) S_val->data;
    h2m_csr_spmm(
        val_type, X0_size, X0_size, Y0_size, row_ptr, col_idx, val, 
        tmpA->data, tmpA->nrow, tmpA1->data, tmpA1->nrow
    );
    // (2.4) Normalize each row of tmpA1
    h2m_2dbuf_resize(QR_buff, sizeof(VT), tmpA1->nrow, 1);
    VT *row_2norm = (VT *) QR_buff->data;
    memset(row_2norm, 0, sizeof(VT) * tmpA1->nrow);
    for (int j = 0; j < tmpA1->ncol; j++)
    {
        VT *tmpA1_j = ((VT *) tmpA1->data) + j * tmpA1->nrow;
        #pragma omp simd 
        for (int i = 0; i < tmpA1->nrow; i++)
            row_2norm[i] += tmpA1_j[i] * tmpA1_j[i];
    }
    for (int i = 0; i < tmpA1->nrow; i++) row_2norm[i] = 1.0 / std::sqrt(row_2norm[i]);
    for (int j = 0; j < tmpA1->ncol; j++)
    {
        VT *tmpA1_j = ((VT *) tmpA1->data) + j * tmpA1->nrow;
        #pragma omp simd 
        for (int i = 0; i < tmpA1->nrow; i++)
            tmpA1_j[i] *= row_2norm[i];
    }
    // (2.5) Compute ID on tmpA1 and select X0 skeleton points to Xp
    int rank = 0, max_rank = 0, n_thread = omp_get_max_threads();
    int *skel_idx = NULL;
    void *V = NULL;
    h2m_2dbuf_resize(ID_buff, sizeof(int), 4 * tmpA1->ncol, 1);
    h2m_2dbuf_resize(QR_buff, sizeof(VT), tmpA1->nrow, tmpA1->ncol);
    id_ppqr(
        tmpA1->nrow, tmpA1->ncol, val_type, tmpA1->data, tmpA1->nrow,
        max_rank, reltol, n_thread, &rank, 
        &skel_idx, &V, ID_buff->data_i, QR_buff->data
    );
    h2m_2dbuf_gather_rows(val_type, X0, Xp, rank, skel_idx);
    free(skel_idx);
    free(V);

    // 3. Select proxy points in domain Y layer by layer
    VT reltol_Yp = (*reltol) * 1e-2;
    if ((val_type == VAL_TYPE_FLOAT) && (reltol_Yp < 1e-6)) reltol_Yp = 1e-6;
    h2m_2dbuf_resize(Yp, sizeof(VT), 0, pt_dim);
    for (int i = 0; i < n_layer; i++)
    {
        // (3.1) Put selected proxy points and i-th layer candidate points together, backup Yp in tmpA
        h2m_2dbuf_resize(tmpA, sizeof(VT), Yp->nrow, pt_dim);
        h2m_2dbuf_resize(Yi, sizeof(VT), Yp->nrow + Y0_lsize, pt_dim);
        VT *Yi_ptr0  = (VT *) Yi->data;
        VT *Yi_ptr1  = Yi_ptr0 + Yp->nrow;
        VT *Y0_i_ptr = ((VT *) Y0_ptr) + i * Y0_lsize;
        // copy_matrix works for row-major matrices
        copy_matrix(sizeof(VT), pt_dim, Yp->nrow, Yp->data, Yp->nrow, tmpA->data, tmpA->nrow, 1);
        copy_matrix(sizeof(VT), pt_dim, Yp->nrow, Yp->data, Yp->nrow, Yi_ptr0, Yi->nrow, 1);
        copy_matrix(sizeof(VT), pt_dim, Y0_lsize, Y0_i_ptr, Y0_size,  Yi_ptr1, Yi->nrow, 1);

        // (3.2) Generate kernel matrix tmpA1 = kernel(Xp, Yi)
        h2m_2dbuf_resize(tmpA1, sizeof(VT), Xp->nrow, Yi->nrow);
        krnl_func_omp(
            Xp->nrow, Xp->nrow, Xp->data, Yi->nrow, Yi->nrow, Yi->data,
            krnl, param, tmpA1->nrow, tmpA1->data, val_type, n_thread
        );

        // (3.3) Compute ID on tmpA1 and select Yi skeleton points to Yp
        h2m_2dbuf_resize(ID_buff, sizeof(int), 4 * tmpA1->ncol, 1);
        h2m_2dbuf_resize(QR_buff, sizeof(VT), tmpA1->nrow, tmpA1->ncol);
        id_ppqr(
            tmpA1->nrow, tmpA1->ncol, val_type, tmpA1->data, tmpA1->nrow,
            max_rank, (const void *) &reltol_Yp, n_thread, &rank, 
            &skel_idx, &V, ID_buff->data_i, QR_buff->data
        );
        h2m_2dbuf_gather_rows(val_type, Yi, Yp, rank, skel_idx);
        free(skel_idx);
        free(V);

        // (3.4) If the new Yp is the same as the old Yp, we can stop here
        if (tmpA->nrow == Yp->nrow)
        {
            VT Yp_err = 0;
            VT *Yp0_data = (VT *) tmpA->data;
            VT *Yp1_data = (VT *) Yp->data;
            for (int k = 0; k < Yp->nrow * pt_dim; k++) 
            {
                VT diff = Yp0_data[k] - Yp1_data[k];
                Yp_err += diff * diff;
            }
            if (Yp_err < reltol1) break;
        }
    }  // End of i loop

    h2m_2dbuf_free(&X0);
    h2m_2dbuf_free(&Xp);
    h2m_2dbuf_free(&Y0);
    h2m_2dbuf_free(&Yi);
    h2m_2dbuf_free(&tmpA);
    h2m_2dbuf_free(&tmpA1);
    h2m_2dbuf_free(&QR_buff);
    h2m_2dbuf_free(&ID_buff);
}

template<typename VT>
static void h2m_octree_proxy_points(
    octree_p octree, krnl_func krnl, const void *param, 
    const void *reltol, h2m_2dbuf_p **pp_
)
{
    int pt_dim     = octree->pt_dim;
    int n_level    = octree->n_level;
    int val_type   = octree->val_type;
    int *ln_displs = octree->ln_displs;
    int *lvl_nodes = octree->lvl_nodes;
    VT  *enbox     = (VT *) octree->enbox;

    h2m_2dbuf_p *pp = (h2m_2dbuf_p *) malloc(sizeof(h2m_2dbuf_p) * n_level);
    for (int i = 0; i < n_level; i++)
        h2m_2dbuf_init(&pp[i], sizeof(VT), 0, pt_dim);

    int root_node = lvl_nodes[ln_displs[0]];
    VT *root_enbox = enbox + root_node * (2 * pt_dim);
    VT root_enbox_size = root_enbox[pt_dim];

    int X0_size   = 2000;
    int Y0_lsize  = 4000;
    int max_layes = 8;
    // Admissible pairs start from level 2
    for (int l = 2; l < n_level; l++)
    {
        int node_l0 = lvl_nodes[ln_displs[l]];
        VT *l0_enbox = enbox + node_l0 * (2 * pt_dim);
        VT L1 = l0_enbox[pt_dim + 0];
        VT L2 = L1 * (1.0 + 2.0 * ALPHA_H2);
        VT L3 = L1 * (1.0 + 8.0 * ALPHA_H2);
        if (L3 > root_enbox_size * 2 - L1) L3 = root_enbox_size * 2.0 - L1;
        h2m_gen_proxy_points<VT>(
            krnl, param, pt_dim, val_type, 
            L1, L2, L3, (VT *) reltol, 
            X0_size, Y0_lsize, max_layes, pp[l]
        );
    }  // End of l loop

    *pp_ = pp;
}

void h2m_octree_proxy_points(
    octree_p octree, krnl_func krnl, const void *param, 
    const void *reltol, h2m_2dbuf_p **lvl_pp 
)
{
    if (octree->val_type == VAL_TYPE_DOUBLE)
        h2m_octree_proxy_points<double>(octree, krnl, param, reltol, lvl_pp);
    if (octree->val_type == VAL_TYPE_FLOAT)
        h2m_octree_proxy_points<float> (octree, krnl, param, reltol, lvl_pp);
}
