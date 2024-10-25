#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cstddef>
#include <cmath>

#include "h2mat_utils.h"
#include "../common.h"

// Initialize a matrix buffer of size nrow * ncol, each data element is of size unit_bytes
void h2m_2dbuf_init(h2m_2dbuf_p *h2m_2dbuf, const size_t unit_bytes, const int nrow, const int ncol)
{
    h2m_2dbuf_p h2m_2dbuf_ = (h2m_2dbuf_p) malloc(sizeof(h2m_2dbuf_s));
    h2m_2dbuf_->data_bytes = unit_bytes * nrow * ncol;
    h2m_2dbuf_->data   = malloc(h2m_2dbuf_->data_bytes);
    h2m_2dbuf_->data_i = (int *) h2m_2dbuf_->data;
    h2m_2dbuf_->nrow = nrow;
    h2m_2dbuf_->ncol = ncol;
    *h2m_2dbuf = h2m_2dbuf_;
}

// Resize a matrix buffer of size nrow * ncol, each data element is of size unit_bytes
void h2m_2dbuf_resize(h2m_2dbuf_p h2m_2dbuf, const size_t unit_bytes, const int nrow, const int ncol)
{
    if (h2m_2dbuf->data_bytes < unit_bytes * nrow * ncol)
    {
        free(h2m_2dbuf->data);
        h2m_2dbuf->data_bytes = unit_bytes * nrow * ncol;
        h2m_2dbuf->data   = malloc(h2m_2dbuf->data_bytes);
        h2m_2dbuf->data_i = (int *) h2m_2dbuf->data;
    }
    h2m_2dbuf->nrow = nrow;
    h2m_2dbuf->ncol = ncol;
}

// Free a h2m_2dbuf
void h2m_2dbuf_free(h2m_2dbuf_p *h2m_2dbuf)
{
    h2m_2dbuf_p h2m_2dbuf_ = *h2m_2dbuf;
    if (h2m_2dbuf_ == NULL) return;
    free(h2m_2dbuf_->data);
    free(h2m_2dbuf_);
    *h2m_2dbuf = NULL;
}

template<typename VT>
static void h2m_2dbuf_gather_rows(h2m_2dbuf_p src, h2m_2dbuf_p dst, const int nrow, const int *row_idx)
{
    int ncol = src->ncol;
    VT *src_data = (VT *) src->data;
    VT *new_data = (VT *) malloc(sizeof(VT) * nrow * ncol);
    for (int j = 0; j < ncol; j++)
    {
        VT *src_j = src_data + j * src->nrow;
        VT *dst_j = new_data + j * nrow;
        #pragma omp simd
        for (int i = 0; i < nrow; i++) dst_j[i] = src_j[row_idx[i]];
    }
    h2m_2dbuf_p dst_ = (dst == NULL) ? src : dst;
    free(dst_->data);
    dst_->data       = new_data;
    dst_->data_i     = (int *) new_data;
    dst_->data_bytes = sizeof(VT) * nrow * ncol;
    dst_->nrow       = nrow;
    dst_->ncol       = ncol;
}

// Gather rows from a h2m_2dbuf to another h2m_2dbuf
void h2m_2dbuf_gather_rows(
    const int val_type, h2m_2dbuf_p src, h2m_2dbuf_p dst, 
    const int nrow, const int *row_idx
)
{
    if (val_type == VAL_TYPE_DOUBLE)
        h2m_2dbuf_gather_rows<double>(src, dst, nrow, row_idx);
    if (val_type == VAL_TYPE_FLOAT)
        h2m_2dbuf_gather_rows<float> (src, dst, nrow, row_idx);
    if (val_type == VAL_TYPE_INT)
        h2m_2dbuf_gather_rows<int>   (src, dst, nrow, row_idx);
}

template<typename VT>
static void h2m_2dbuf_gather_cols(h2m_2dbuf_p src, h2m_2dbuf_p dst, const int ncol, const int *col_idx)
{
    int nrow = src->nrow;
    VT *src_data = (VT *) src->data;
    VT *new_data = (VT *) malloc(sizeof(VT) * nrow * ncol);
    for (int j = 0; j < ncol; j++)
    {
        VT *src_j = src_data + col_idx[j] * nrow;
        VT *dst_j = new_data + j * nrow;
        memcpy(dst_j, src_j, sizeof(VT) * nrow);
    }
    h2m_2dbuf_p dst_ = (dst == NULL) ? src : dst;
    free(dst->data);
    dst_->data   = new_data;
    dst_->data_i = (int *) new_data;
    dst_->nrow   = nrow;
    dst_->ncol   = ncol;
}

// Gather columns from a h2m_2dbuf to another h2m_2dbuf
void h2m_2dbuf_gather_cols(
    const int val_type, h2m_2dbuf_p src, h2m_2dbuf_p dst, 
    const int ncol, const int *col_idx
)
{
    if (val_type == VAL_TYPE_DOUBLE)
        h2m_2dbuf_gather_cols<double>(src, dst, ncol, col_idx);
    if (val_type == VAL_TYPE_FLOAT)
        h2m_2dbuf_gather_cols<float> (src, dst, ncol, col_idx);
    if (val_type == VAL_TYPE_INT)
        h2m_2dbuf_gather_cols<int>   (src, dst, ncol, col_idx);
}

template<typename VT>
static int h2m_is_point_in_enbox(const int dim, const VT *x, const VT *enbox)
{
    for (int i = 0; i < dim; i++)
        if ((x[i] < enbox[i]) || (x[i] > enbox[i] + enbox[i + dim])) return 0;
    return 1;
}

// Check if a point is in an enclosing box
int h2m_is_point_in_enbox(const int val_type, const int dim, const void *x, const void *enbox)
{
    int ret = 0;
    if (val_type == VAL_TYPE_DOUBLE) 
        ret = h2m_is_point_in_enbox<double>(dim, (const double *) x, (const double *) enbox);
    if (val_type == VAL_TYPE_FLOAT)  
        ret = h2m_is_point_in_enbox<float> (dim, (const float *)  x, (const float *)  enbox);
    return ret;
}

template<typename VT>
int h2m_is_admissible_enbox_pair(const int dim, const VT *enbox0, const VT *enbox1)
{
    for (int i = 0; i < dim; i++)
    {
        VT width0  = enbox0[dim + i];
        VT width1  = enbox1[dim + i];
        VT center0 = enbox0[i] + width0 * 0.5;
        VT center1 = enbox1[i] + width1 * 0.5;
        VT min_w   = (width0 < width1) ? width0 : width1;
        VT dist    = std::abs(center0 - center1);
        if (dist >= ALPHA_H2 * min_w + 0.5 * (width0 + width1)) return 1;
    }
    return 0;
}

// Check if two enclosing boxes are admissible
int h2m_is_admissible_enbox_pair(const int val_type, const int dim, const void *enbox0, const void *enbox1)
{
    int ret = 0;
    if (val_type == VAL_TYPE_DOUBLE) 
        ret = h2m_is_admissible_enbox_pair<double>(dim, (const double *) enbox0, (const double *) enbox1);
    if (val_type == VAL_TYPE_FLOAT)
        ret = h2m_is_admissible_enbox_pair<float> (dim, (const float *)  enbox0, (const float *)  enbox1);
    return ret;
}

template<typename VT>
void h2m_rand_points_in_shell(const int n, const int dim, const VT *L0, const VT *L1, VT *x, const int ldx)
{
    int cnt = 0;
    VT L0_ = *L0, L1_ = *L1;
    VT semi_L0 = L0_ / 2, semi_L1 = L1_ / 2;
    VT *coord = (VT *) malloc(sizeof(VT) * dim);
    while (cnt < n)
    {
        int in_L0 = 1;
        for (int i = 0; i < dim; i++)
        {
            coord[i] = L1_ * ((VT) rand() / (VT) RAND_MAX) - semi_L1;
            if ((coord[i] < -semi_L0) || (coord[i] > semi_L0)) in_L0 = 0;
        }
        if (in_L0) continue;
        for (int i = 0; i < dim; i++) x[cnt + i * ldx] = coord[i];
        cnt++;
    }
    free(coord);
}

// Generate n uniformly distributed random points in a shell
// [-L1/2, L1/2]^dim excluding [-L0/2, L0/2]^dim
void h2m_rand_points_in_shell(
    const int val_type, const int n, const int dim, const void *L0, 
    const void *L1, void *x, const int ldx
)
{
    if (val_type == VAL_TYPE_DOUBLE)
        h2m_rand_points_in_shell<double>(n, dim, (const double *) L0, (const double *) L1, (double *) x, ldx);
    if (val_type == VAL_TYPE_FLOAT)
        h2m_rand_points_in_shell<float> (n, dim, (const float *)  L0, (const float *)  L1, (float *)  x, ldx);
}

// Randomly sample k different integers in [0, n-1]
// Input parameters:
//   n, k : Sample k integers in [0, n-1]
//   work : Optional, size n bytes, work buffer, can be NULL
// Output parameter:
//   samples : Size k, sampled integers
void h2m_rand_sample(const int n, const int k, int *samples, void *work)
{
    // TODO: replace this with a bitmap implementation
    uint8_t *flag = NULL;
    if (work != NULL) flag = (uint8_t *) work;
    else flag = (uint8_t *) malloc(n);
    memset(flag, 0, n);
    for (int i = 0; i < k; i++)
    {
        int idx = rand() % n;
        while (flag[idx] == 1) idx = rand() % n;
        samples[i] = idx;
        flag[idx] = 1;
    }
    if (work == NULL) free(flag);
}

template<typename VT>
void h2m_sub_gaussian_csr(
    const int nrow, const int ncol, const int max_nnz_row, 
    h2m_2dbuf_p idx, h2m_2dbuf_p val
)
{
    int nnz_row_ = (max_nnz_row < ncol) ? max_nnz_row : ncol;
    int nnz = nrow * nnz_row_;
    h2m_2dbuf_resize(idx, sizeof(int), nrow + 1 + nnz + ncol, 1);
    h2m_2dbuf_resize(val, sizeof(VT),  nnz, 1);
    h2m_2dbuf_ivec_set_size(idx, nrow + 1 + nnz);  // The last ncol elements are work buffer
    int *row_ptr = idx->data_i;
    int *col_idx = row_ptr + (nrow + 1);
    int *flag    = col_idx + nnz;
    VT  *val_    = (VT *) val->data;
    for (int i = 0; i < nnz; i++) val_[i] = (VT) (2.0 * (rand() & 1) - 1.0);
    for (int i = 0; i < nrow; i++)
    {
        row_ptr[i] = i * nnz_row_;
        int *row_i_cols = col_idx + i * nnz_row_;
        h2m_rand_sample(ncol, nnz_row_, row_i_cols, flag);
    }
    row_ptr[nrow] = nnz;
}

// Generate a sub-Gaussian random sparse CSR matrix with a fixed nnz 
// per row, each nonzero entry is +1 or -1 with equal probability
void h2m_sub_gaussian_csr(
    const int val_type, const int nrow, const int ncol, 
    const int max_nnz_row, h2m_2dbuf_p idx, h2m_2dbuf_p val
)
{
    if (val_type == VAL_TYPE_DOUBLE)
        h2m_sub_gaussian_csr<double>(nrow, ncol, max_nnz_row, idx, val);
    if (val_type == VAL_TYPE_FLOAT)
        h2m_sub_gaussian_csr<float> (nrow, ncol, max_nnz_row, idx, val);
}

template<typename VT>
void h2m_csr_spmm(
    const int m, const int n, const int k,
    const int *row_ptr, const int *col_idx, const VT *val,
    const VT *B, const int ldB, VT *C, const int ldC
)
{
    // Since the CSR matrix we will use here is generated by h2m_sub_gaussian_csr(), 
    // using a naive OpenMP CSR SpMM is good enough at the moment
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < n; j++)
        {
            VT sum = 0.0;
            const VT *Bj = B + j * ldB;
            #pragma omp simd
            for (int k = row_ptr[i]; k < row_ptr[i + 1]; k++)
                sum += val[k] * Bj[col_idx[k]];
            C[i + j * ldC] = sum;
        }
    }
}

// Compute SpMM C = A * B, where A is a CSR matrix, B and C are dense matrices
void h2m_csr_spmm(
    const int val_type, const int m, const int n, const int k,
    const int *row_ptr, const int *col_idx, const void *val,
    const void *B, const int ldB, void *C, const int ldC
)
{
    if (val_type == VAL_TYPE_DOUBLE)
    {
        h2m_csr_spmm<double>(
            m, n, k, row_ptr, col_idx, (const double *) val, 
            (const double *) B, ldB, (double *) C, ldC
        );
    }
    if (val_type == VAL_TYPE_FLOAT)
    {
        h2m_csr_spmm<float>(
            m, n, k, row_ptr, col_idx, (const float *)  val, 
            (const float *)  B, ldB, (float *)  C, ldC
        );
    }
}
