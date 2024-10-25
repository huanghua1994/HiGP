#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <complex.h>
#include <sys/time.h>
#include <math.h>

#include "utils.h"

// Get wall-clock time in seconds
double get_wtime_sec()
{
    double sec;
    struct timeval tv;
    gettimeofday(&tv, NULL);
    sec = tv.tv_sec + (double) tv.tv_usec / 1000000.0;
    return sec;
}

// Partition an array into multiple same-size blocks and return the 
// start position of a given block
void calc_block_spos_len(
    const int len, const int nblk, const int iblk,
    int *blk_spos, int *blk_len
)
{
	if (iblk < 0 || iblk > nblk)
    {
        *blk_spos = -1;
        *blk_len  = 0;
        return;
    }
	int rem = len % nblk;
	int bs0 = len / nblk;
	int bs1 = bs0 + 1;
	if (iblk < rem) 
    {
        *blk_spos = bs1 * iblk;
        *blk_len  = bs1;
    } else {
        *blk_spos = bs0 * iblk + rem;
        *blk_len  = bs0;
    }
}

// Allocate a piece of aligned memory 
void *malloc_aligned(size_t size, size_t alignment)
{
    void *ptr = NULL;
    posix_memalign(&ptr, alignment, size);
    return ptr;
}

// Free a piece of aligned memory allocated by malloc_aligned()
void free_aligned(void *mem)
{
    free(mem);
}

// Calculate the 2-norm of a vector
// Warning: this is a naive implementation, not numerically stable
void calc_2norm(utils_dtype_t dtype, const int len, const void *x, void *ret)
{
    if (dtype == UTILS_DTYPE_FP64)
    {
        double res = 0.0;
        double *x_ = (double *) x;
        for (int i = 0; i < len; i++) res += x_[i] * x_[i];
        *((double *) ret) = sqrt(res);
    }
    if (dtype == UTILS_DTYPE_FP32)
    {
        float res = 0.0;
        float *x_ = (float *) x;
        for (int i = 0; i < len; i++) res += x_[i] * x_[i];
        *((float *) ret) = sqrtf(res);
    }
}

// Calculate the 2-norm of the difference between two vectors 
// and the 2-norm of the reference vector 
void calc_err_2norm(
    utils_dtype_t dtype, const int len, const void *x0, const void *x1, 
    void *x0_2norm_, void *err_2norm_
)
{
    if (dtype == UTILS_DTYPE_FP64)
    {
        double x0_2norm = 0.0, err_2norm = 0.0, diff;
        double *x0_ = (double *) x0;
        double *x1_ = (double *) x1;
        for (int i = 0; i < len; i++)
        {
            diff = x0_[i] - x1_[i];
            x0_2norm  += x0_[i] * x0_[i];
            err_2norm += diff  * diff;
        }
        *((double *) x0_2norm_)  = sqrt(x0_2norm);
        *((double *) err_2norm_) = sqrt(err_2norm);
    }
    if (dtype == UTILS_DTYPE_FP32)
    {
        float x0_2norm = 0.0, err_2norm = 0.0, diff;
        float *x0_ = (float *) x0;
        float *x1_ = (float *) x1;
        for (int i = 0; i < len; i++)
        {
            diff = x0_[i] - x1_[i];
            x0_2norm  += x0_[i] * x0_[i];
            err_2norm += diff  * diff;
        }
        *((float *) x0_2norm_)  = sqrtf(x0_2norm);
        *((float *) err_2norm_) = sqrtf(err_2norm);
    }
}

// Copy a row-major matrix to another row-major matrix
void copy_matrix(
    const size_t dt_size, const int nrow, const int ncol, 
    const void *src, const int lds, void *dst, const int ldd, const int use_omp
)
{
    const char *src_ = (char*) src;
    char *dst_ = (char*) dst;
    const size_t lds_ = dt_size * (size_t) lds;
    const size_t ldd_ = dt_size * (size_t) ldd;
    const size_t row_msize = dt_size * (size_t) ncol;
    if (use_omp == 0)
    {
        for (int irow = 0; irow < nrow; irow++)
        {
            size_t src_offset = (size_t) irow * lds_;
            size_t dst_offset = (size_t) irow * ldd_;
            memcpy(dst_ + dst_offset, src_ + src_offset, row_msize);
        }
    } else {
        #if defined(_OPENMP)
        #pragma omp parallel for schedule(static)
        #endif
        for (int irow = 0; irow < nrow; irow++)
        {
            size_t src_offset = (size_t) irow * lds_;
            size_t dst_offset = (size_t) irow * ldd_;
            memcpy(dst_ + dst_offset, src_ + src_offset, row_msize);
        }
    }
}

// Gather elements from a vector to another vector
void gather_vector_elements(const size_t dt_size, const int nelem, const int *idx, const void *src, void *dst)
{
    if (dt_size == 4)
    {
        const float *src_ = (float*) src;
        float *dst_ = (float*) dst;
        #if defined(_OPENMP)
        #pragma omp simd
        #endif
        for (int i = 0; i < nelem; i++) dst_[i] = src_[idx[i]];
    }
    if (dt_size == 8)
    {
        const double *src_ = (double*) src;
        double *dst_ = (double*) dst;
        #if defined(_OPENMP)
        #pragma omp simd
        #endif
        for (int i = 0; i < nelem; i++) dst_[i] = src_[idx[i]];
    }
    if (dt_size == 16)
    {
        const double _Complex *src_ = (double _Complex*) src;
        double _Complex *dst_ = (double _Complex*) dst;
        #if defined(_OPENMP)
        #pragma omp simd
        #endif
        for (int i = 0; i < nelem; i++) dst_[i] = src_[idx[i]];
    }
}

// Gather rows from a matrix to another matrix
void gather_matrix_rows(
    const size_t dt_size, const int nrow, const int ncol, const int *idx, 
    const void *src, const int lds, void *dst, const int ldd
)
{
    const char *src_ = (char*) src;
    char *dst_ = (char*) dst;
    const size_t lds_ = dt_size * (size_t) lds;
    const size_t ldd_ = dt_size * (size_t) ldd;
    const size_t row_msize = dt_size * (size_t) ncol;
    #if defined(_OPENMP)
    #pragma omp parallel for schedule(static)
    #endif
    for (int irow = 0; irow < nrow; irow++)
    {
        size_t src_offset = (size_t) idx[irow] * lds_;
        size_t dst_offset = (size_t) irow * ldd_;
        memcpy(dst_ + dst_offset, src_ + src_offset, row_msize);
    }
}

// Gather columns from a matrix to another matrix
void gather_matrix_cols(
    const size_t dt_size, const int nrow, const int ncol, const int *idx, 
    const void *src, const int lds, void *dst, const int ldd
)
{
    const char *src_ = (char*) src;
    char *dst_ = (char*) dst;
    const size_t lds_ = dt_size * (size_t) lds;
    const size_t ldd_ = dt_size * (size_t) ldd;
    #if defined(_OPENMP)
    #pragma omp parallel for schedule(static)
    #endif
    for (int irow = 0; irow < nrow; irow++)
    {
        size_t src_offset = (size_t) irow * lds_;
        size_t dst_offset = (size_t) irow * ldd_;
        gather_vector_elements(dt_size, ncol, idx, src_ + src_offset, dst_ + dst_offset);
    }
}

// Print a matrix to standard output
void print_matrix(
    utils_dtype_t dtype, const int stype, const void *mat, const int ldm, 
    const int nrow, const int ncol, const char *fmt, const char *name
)
{
    printf("%s:\n", name);
    int row_stride, col_stride;
    if (stype == 0)
    {
        row_stride = ldm;
        col_stride = 1;
    } else {
        row_stride = 1;
        col_stride = ldm;
    }

    #define PRINT_MATRIX(DTYPE) \
    do { \
        const DTYPE *mat_ = (const DTYPE *) mat; \
        for (int i = 0; i < nrow; i++) \
        { \
            for (int j = 0; j < ncol; j++) \
            { \
                printf(fmt, mat_[i * row_stride + j * col_stride]); \
            } \
            printf("\n"); \
        } \
    } while (0)

    if (dtype == UTILS_DTYPE_INT32) PRINT_MATRIX(int);
    if (dtype == UTILS_DTYPE_FP64)  PRINT_MATRIX(double);
    if (dtype == UTILS_DTYPE_FP32)  PRINT_MATRIX(float);

    #undef PRINT_MATRIX
}

// Dump binary to file
void dump_binary(const char *fname, void *data, const size_t bytes)
{
    FILE *fp = fopen(fname, "wb");
    fwrite(data, 1, bytes, fp);
    fclose(fp);
}

// Read binary from file
void read_binary_file(const char *fname, void *data, const size_t bytes)
{
    FILE *fp = fopen(fname, "rb");
    fread(data, 1, bytes, fp);
    fclose(fp);
}
