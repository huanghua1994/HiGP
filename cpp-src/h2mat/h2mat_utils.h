#ifndef __H2MAT_UTILS_H__
#define __H2MAT_UTILS_H__

#include <stdlib.h>
#include <string.h>
#include "../utils.h"

#define ALPHA_H2 0.99

struct h2m_2dbuf
{
    void   *data;       // Pointer to the data buffer. If used as a matrix, data is col-major.
    int    *data_i;     // (int *) data
    size_t data_bytes;  // Size of the data buffer in bytes
    int nrow, ncol;     // Number of rows and columns. If used as a vector, only nrow is used.
};
typedef struct h2m_2dbuf  h2m_2dbuf_s;
typedef struct h2m_2dbuf *h2m_2dbuf_p;

#ifdef __cplusplus
extern "C" {
#endif

// Initialize a matrix buffer of size nrow * ncol, each data element is of size unit_bytes
void h2m_2dbuf_init(h2m_2dbuf_p *h2m_2dbuf, const size_t unit_bytes, const int nrow, const int ncol);

// Resize a matrix buffer of size nrow * ncol, each data element is of size unit_bytes.
// If the resized buffer is smaller than the original one, h2m_2dbuf->data will not be touched, 
// only h2m_2dbuf->{nrow, ncol} will be updated. 
// If the resized buffer is larger than the original one, h2m_2dbuf->data will be reallocated 
// and the original data may not be preserved.
void h2m_2dbuf_resize(h2m_2dbuf_p h2m_2dbuf, const size_t unit_bytes, const int nrow, const int ncol);

// Free a h2m_2dbuf
void h2m_2dbuf_free(h2m_2dbuf_p *h2m_2dbuf);

// Push a int value to the end of a h2m_debuf
static void h2m_2dbuf_ivec_push(h2m_2dbuf_p h2m_2dbuf, const int val)
{
    if (h2m_2dbuf->nrow * sizeof(int) >= h2m_2dbuf->data_bytes)
    {
        size_t new_data_bytes = sizeof(int) * h2m_2dbuf->nrow * 2;
        void *new_data = malloc(new_data_bytes);
        memcpy(new_data, h2m_2dbuf->data, sizeof(int) * h2m_2dbuf->nrow);
        free(h2m_2dbuf->data);
        h2m_2dbuf->data   = new_data;
        h2m_2dbuf->data_i = (int *) new_data;
        h2m_2dbuf->data_bytes = new_data_bytes;
    }
    h2m_2dbuf->data_i[h2m_2dbuf->nrow++] = val;
}

// Set the integer vector size of a h2m_debuf
static void h2m_2dbuf_ivec_set_size(h2m_2dbuf_p h2m_2dbuf, const int size)
{
    assert(size * sizeof(int) <= h2m_2dbuf->data_bytes);
    h2m_2dbuf->nrow = size;
}

static int h2m_2dbuf_ivec_get_size(h2m_2dbuf_p h2m_2dbuf)
{
    return h2m_2dbuf->nrow;
}

// Gather rows from a h2m_2dbuf to another h2m_2dbuf
// Input parameters:
//   val_type : Data type of src and dst, 0 for double, 1 for float, 2 for int
//   src      : Source h2m_2dbuf
//   nrow     : Number of rows to gather
//   row_idx  : Row indices to gather, size nrow
// Output parameter:
//   dst : Destination h2m_2dbuf. If dst == NULL, src will be used as dst.
void h2m_2dbuf_gather_rows(
    const int val_type, h2m_2dbuf_p src, h2m_2dbuf_p dst, 
    const int nrow, const int *row_idx
);

// Gather columns from a h2m_2dbuf to another h2m_2dbuf
// Input parameters:
//   val_type : Data type of src and dst, 0 for double, 1 for float, 2 for int
//   src      : Source h2m_2dbuf
//   ncol     : Number of rows to gather
//   col_idx  : Column indices to gather, size ncol
// Output parameter:
//   dst : Destination h2m_2dbuf. If dst == NULL, src will be used as dst.
void h2m_2dbuf_gather_cols(
    const int val_type, h2m_2dbuf_p src, h2m_2dbuf_p dst, 
    const int ncol, const int *col_idx
);

// Check if a point is in an enclosing box
// Input parameters:
//   val_type : Data type of x and enbox, 0 for double, 1 for float
//   dim      : Dimension of x and enbox
//   x        : Point coordinates, size dim
//   enbox    : Enclosing box, size dim * 2, first dim elements are the lower bounds,
//              last dim elements are the size of the box in each dimension
int h2m_is_point_in_enbox(const int val_type, const int dim, const void *x, const void *enbox);

// Check if two enclosing boxes are admissible
// Input parameters:
//   val_type : Data type of enbox0 and enbox1, 0 for double, 1 for float
//   dim      : Dimension of enbox0 and enbox1
//   enbox0   : First enclosing box, size dim * 2
//   enbox1   : Second enclosing box, size dim * 2
int h2m_is_admissible_enbox_pair(const int val_type, const int dim, const void *enbox0, const void *enbox1);

// Generate n uniformly distributed random points in a shell
// [-L1/2, L1/2]^dim excluding [-L0/2, L0/2]^dim
// Input parameters:
//   val_type : Data type of L0, L1, and x, 0 for double, 1 for float
//   n, dim   : Number of points to generate and point dimension
//   L0, L1   : Inner and outer shell size
//   ldx      : Leading dimension of x, >= n
// Output parameters:
//   x : Generated points, size ldx * dim, col-major, each row is a point
void h2m_rand_points_in_shell(
    const int val_type, const int n, const int dim, const void *L0, 
    const void *L1, void *x, const int ldx
);

// Generate a sub-Gaussian random sparse CSR matrix with a fixed nnz 
// per row, each nonzero entry is +1 or -1 with equal probability
// Input parameters:
//   val_type    : Data type of val, 0 for double, 1 for float
//   nrow, ncol  : Size of the sparse matrix
//   max_nnz_row : Maximum number of nonzero entries per row
// Output parameters:
//   idx : CSR row_ptr and col_idx, size nrow + 1 + nnz (idx->nrow)
//   val : CSR values, size nnz (val->nrow)
void h2m_sub_gaussian_csr(
    const int val_type, const int nrow, const int ncol, 
    const int max_nnz_row, h2m_2dbuf_p idx, h2m_2dbuf_p val
);

// Compute SpMM C = A * B, where A is a CSR matrix, B and C are dense matrices
// Input parameters:
//   val_type : Data type of val, B, and C, 0 for double, 1 for float
//   m, n, k  : A is m * k, B is k * n, C is m * n
//   row_ptr  : CSR row_ptr, size m + 1
//   col_idx  : CSR col_idx, size row_ptr[m]
//   val      : CSR values, size row_ptr[m]
//   B        : Dense matrix B, size ldb * n, col-major
//   ldB      : Leading dimension of B, >= k
//   ldC      : Leading dimension of C, >= m
// Output parameter:
//   C : Dense matrix C, size ldC * n, col-major
void h2m_csr_spmm(
    const int val_type, const int m, const int n, const int k,
    const int *row_ptr, const int *col_idx, const void *val,
    const void *B, const int ldB, void *C, const int ldC
);

#ifdef __cplusplus
};
#endif

#endif
