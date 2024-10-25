#ifndef __CPU_LINALG_ILP64_H__
#define __CPU_LINALG_ILP64_H__

// BLAS & LAPACK ILP64 ABI: interger, long and pointer are 64-bit
// We have seen this on OpenBLAS. MKL ILP64 does not have the 64_ suffix.

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// ========== BLAS ========== //

float  snrm2_64_(const int64_t *n, const float  *x, const int64_t *incx);
double dnrm2_64_(const int64_t *n, const double *x, const int64_t *incx);

float  sdot_64_(const int64_t *n, const float  *x, const int64_t *incx, const float  *y, const int64_t *incy);
double ddot_64_(const int64_t *n, const double *x, const int64_t *incx, const double *y, const int64_t *incy);

int sgemv_64_(
    const char *trans, const int64_t *m, const int64_t *n,
    const float *alpha, const float *a, const int64_t *lda, const float *x, const int64_t *incx,
    const float *beta, float *y, const int64_t *incy
);
int dgemv_64_(
    const char *trans, const int64_t *m, const int64_t *n,
    const double *alpha, const double *a, const int64_t *lda, const double *x, const int64_t *incx,
    const double *beta, double *y, const int64_t *incy
);

int sgemm_64_(
    const char *transa, const char *transb, const int64_t *m, const int64_t *n, const int64_t *k,
    const float *alpha, const float *a, const int64_t *lda, const float *b, const int64_t *ldb,
    const float *beta, float *c, const int64_t *ldc
);
int dgemm_64_(
    const char *transa, const char *transb, const int64_t *m, const int64_t *n, const int64_t *k,
    const double *alpha, const double *a, const int64_t *lda, const double *b, const int64_t *ldb,
    const double *beta, double *c, const int64_t *ldc
);

int strsm_64_(
    const char *side, const char *uplo, const char *transa, const char *diag, const int64_t *m, const int64_t *n, 
    const float  *alpha, const float  *a, const int64_t *lda, float  *b, const int64_t *ldb
);
int dtrsm_64_(
    const char *side, const char *uplo, const char *transa, const char *diag, const int64_t *m, const int64_t *n, 
    const double *alpha, const double *a, const int64_t *lda, double *b, const int64_t *ldb
);

int ssyrk_64_(
    const char *uplo, const char *trans, const int64_t *n, const int64_t *k, const float  *alpha, 
    const float  *a, const int64_t *lda, const float  *beta, float  *c, const int64_t *ldc
);
int dsyrk_64_(
    const char *uplo, const char *trans, const int64_t *n, const int64_t *k, const double *alpha, 
    const double *a, const int64_t *lda, const double *beta, double *c, const int64_t *ldc
);

// ========== LAPACK ========== //

void spotrf_64_(const char *uplo, const int64_t *n, float  *a, const int64_t *lda, int64_t *info);
void dpotrf_64_(const char *uplo, const int64_t *n, double *a, const int64_t *lda, int64_t *info);

void spotrs_64_(
    const char *uplo, const int64_t *n, const int64_t *nrhs, float  *a, const int64_t *lda, 
    float  *b, const int64_t *ldb, int64_t *info
);
void dpotrs_64_(
    const char *uplo, const int64_t *n, const int64_t *nrhs, double *a, const int64_t *lda, 
    double *b, const int64_t *ldb, int64_t *info
);

void sgesv_64_(
    const int64_t *n, const int64_t *nrhs, float  *a, const int64_t *lda, int64_t *ipiv, 
    float  *b, const int64_t *ldb, int64_t *info
);
void dgesv_64_(
    const int64_t *n, const int64_t *nrhs, double *a, const int64_t *lda, int64_t *ipiv, 
    double *b, const int64_t *ldb, int64_t *info
);

void sposv_64_(
    const char *uplo, const int64_t *n, const int64_t *nrhs, float  *a, const int64_t *lda, 
    float  *b, const int64_t *ldb, int64_t *info
);
void dposv_64_(
    const char *uplo, const int64_t *n, const int64_t *nrhs, double *a, const int64_t *lda, 
    double *b, const int64_t *ldb, int64_t *info
);

void ssyev_64_(
    const char *jobz, const char *uplo, const int64_t *n, float  *a, const int64_t *lda,
    float  *w, float  *work, const int64_t *lwork, int64_t *info
);
void dsyev_64_(
    const char *jobz, const char *uplo, const int64_t *n, double *a, const int64_t *lda,
    double *w, double *work, const int64_t *lwork, int64_t *info
);

void ssyevd_64_(
    const char *jobz, const char *uplo, const int64_t *n, float  *a, const int64_t *lda,
    float  *w, float  *work, const int64_t *lwork, int64_t *iwork, const int64_t *liwork, int64_t *info
);
void dsyevd_64_(
    const char *jobz, const char *uplo, const int64_t *n, double *a, const int64_t *lda,
    double *w, double *work, const int64_t *lwork, int64_t *iwork, const int64_t *liwork, int64_t *info
);

void sgesvd_64_(
    const char *jobu, const char *jobvt, const int64_t *m, const int64_t *n, 
    float  *a, const int64_t *lda, float  *s, float  *u, const int64_t *ldu, 
    float  *vt, const int64_t *ldvt, float  *work, const int64_t *lwork, int64_t *info
);
void dgesvd_64_(
    const char *jobu, const char *jobvt, const int64_t *m, const int64_t *n, 
    double *a, const int64_t *lda, double *s, double *u, const int64_t *ldu, 
    double *vt, const int64_t *ldvt, double *work, const int64_t *lwork, int64_t *info
);

#ifdef __cplusplus
}
#endif

static const char * const notrans = "N";
static const char * const trans   = "T";
static const char * const upper   = "U";
static const char * const lower   = "L";
static const char * const left    = "L";
static const char * const right   = "R";
static const char * const nonunit = "N";
static const char * const unit    = "U";
static const char * const vector  = "V";
static const char * const overwrt = "O";
static const char * const nocalc  = "N";

#endif  // __CPU_LINALG_ILP64_H__
