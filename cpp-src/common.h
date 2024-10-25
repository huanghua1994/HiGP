#ifndef __COMMON_H__
#define __COMMON_H__

typedef enum val_type
{
    VAL_TYPE_DOUBLE = 0,
    VAL_TYPE_FLOAT  = 1,
    VAL_TYPE_INT    = 2
} val_type_t;

typedef enum symm_kmat_alg
{
    SYMM_KMAT_ALG_DENSE = 0,        // dense_krnl_mat_p, on-the-fly matmul
    SYMM_KMAT_ALG_DENSE_FORM = 1,   // Form the full matrix / fall back to on-the-fly 
    SYMM_KMAT_ALG_H2_PROXY = 2      // ss_h2mat_p, using proxy points
} symm_kmat_alg_t;

typedef enum gp_task
{
    GP_REGRESSION = 0,
    GP_CLASSIFICATION = 1
} gp_task_t;

#ifdef __cplusplus
extern "C" {
#endif

// Interface for evaluating a kernel matrix
// Input parameters:
//   n{0, 1}      : Numer of of points in the 1st and 2nd point sets
//   ld{0, 1}     : Leading dimension of c{0, 1}, >= n{0, 1}
//   c{0, 1}      : Point coordinates, col-major, size ld{0, 1} * dim, each row is a point
//   param        : Paramater list, [dim, <other parameters>]
//   ldm          : >= n0, leading dimension of mat
//   val_type     : Data type of c0, c1, param, and mat, 0 for double, 1 for float
// Output parameters:
//   mat : Size ldm * n1, col-major, K(c0, c1)
typedef void (*krnl_func) (
    const int n0, const int ld0, const void *c0,
    const int n1, const int ld1, const void *c1,
    const void *param, const int ldm, void *mat, const int val_type
);

// Interface for evaluating a kernel matrix and/or its derivate w.r.t. parameter l
// Input parameters:
//   n{0, 1}      : Numer of of points in the 1st and 2nd point sets
//   ld{0, 1}     : Leading dimension of c{0, 1}, >= n{0, 1}
//   c{0, 1}      : Point coordinates, col-major, size ld{0, 1} * dim, each row is a point
//   param        : Paramater list, [dim, l]
//   ldm          : >= n0, leading dimension of mat
//   val_type     : Data type of c0, c1, param, k_mat, and dl_mat,
//                  0 for double, 1 for float
//   require_krnl : 0 or 1, if K(c0, c1) is required
//   require_grad : 0 or 1, if dK(c0, c1) / dl is required
// Output parameters:
//   k_mat  : Size ldm * n1, col-major, K(c0, c1)
//   dl_mat : Size ldm * n1, col-major, dK(c0, c1) / dl
// Note:
//  1. If require_{krnl, grad} == 0, {k, dl}_mat will not be referenced
//  2. require_{krnl, grad} cannot be both of 0
typedef void (*krnl_grad_func) (
    const int n0, const int ld0, const void *c0,
    const int n1, const int ld1, const void *c1,
    const void *param, const int ldm, const int val_type, 
    const int require_krnl, void *k_mat, 
    const int require_grad, void *dl_mat
);

// C := A * B
// Input parameters:
//   A   : Pointer to a data structure that represents a matrix A of size m * k
//   n   : Number of columns in matrices B and C
//   B   : Size ldB * n, col-major dense input matrix
//   ldB : Leading dimension of B, >= k
//   ldC : Leading dimension of C, >= m
// Output parameters:
//   C : Size ldC * n, col-major dense result matrix
typedef void (*matmul_fptr) (const void *A, const int n, const void *B, const int ldB, const void *C, const int ldC);

#ifdef __cplusplus
}  // extern "C"
#endif


#ifdef __cplusplus
// Quick sort (ascending) for (KT, VT) key-value pairs
template<typename KT, typename VT>
static void qsort_key_val(KT *key, VT *val, const int l, const int r)
{
    int i = l, j = r;
    const KT mid_key = key[(l + r) / 2];
    while (i <= j)
    {
        while (key[i] < mid_key) i++;
        while (key[j] > mid_key) j--;
        if (i <= j)
        {
            KT tmp_key = key[i]; key[i] = key[j];  key[j] = tmp_key;
            VT tmp_val = val[i]; val[i] = val[j];  val[j] = tmp_val;
            i++;  j--;
        }
    }
    if (i < r) qsort_key_val<KT, VT>(key, val, i, r);
    if (j > l) qsort_key_val<KT, VT>(key, val, l, j);
}

// Quick partitioning for (KT, VT) key-value pairs and get the first k smallest elements
template<typename KT, typename VT>
static void qpart_key_val(KT *key, VT *val, const int l, const int r, const int k)
{
    int i = l, j = r;
    const KT mid_key = key[(l + r) / 2];
    while (i <= j)
    {
        while (key[i] < mid_key) i++;
        while (key[j] > mid_key) j--;
        if (i <= j)
        {
            KT tmp_key = key[i]; key[i] = key[j];  key[j] = tmp_key;
            VT tmp_val = val[i]; val[i] = val[j];  val[j] = tmp_val;
            i++;  j--;
        }
    }
    if (j > l) qpart_key_val<KT, VT>(key, val, l, j, k);
    if ((i < r) && (i < k)) qpart_key_val<KT, VT>(key, val, i, r, k);
}
#endif

#endif
