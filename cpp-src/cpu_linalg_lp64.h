#ifndef __CPU_LINALG_LP64_H__
#define __CPU_LINALG_LP64_H__

// BLAS & LAPACK LP64 ABI: interger is 32-bit, long and pointer are 64-bit

#ifdef __cplusplus
extern "C" {
#endif

// ========== BLAS ========== //

float  snrm2_(const int *n, const float  *x, const int *incx);
double dnrm2_(const int *n, const double *x, const int *incx);

float  sdot_(const int *n, const float  *x, const int *incx, const float  *y, const int *incy);
double ddot_(const int *n, const double *x, const int *incx, const double *y, const int *incy);

int sgemv_(
    const char *trans, const int *m, const int *n,
    const float *alpha, const float *a, const int *lda, const float *x, const int *incx,
    const float *beta, float *y, const int *incy
);
int dgemv_(
    const char *trans, const int *m, const int *n,
    const double *alpha, const double *a, const int *lda, const double *x, const int *incx,
    const double *beta, double *y, const int *incy
);

int sgemm_(
    const char *transa, const char *transb, const int *m, const int *n, const int *k,
    const float *alpha, const float *a, const int *lda, const float *b, const int *ldb,
    const float *beta, float *c, const int *ldc
);
int dgemm_(
    const char *transa, const char *transb, const int *m, const int *n, const int *k,
    const double *alpha, const double *a, const int *lda, const double *b, const int *ldb,
    const double *beta, double *c, const int *ldc
);

int strsm_(
    const char *side, const char *uplo, const char *transa, const char *diag, const int *m, const int *n, 
    const float  *alpha, const float  *a, const int *lda, float  *b, const int *ldb
);
int dtrsm_(
    const char *side, const char *uplo, const char *transa, const char *diag, const int *m, const int *n, 
    const double *alpha, const double *a, const int *lda, double *b, const int *ldb
);

int ssyrk_(
    const char *uplo, const char *trans, const int *n, const int *k, const float  *alpha, 
    const float  *a, const int *lda, const float  *beta, float  *c, const int *ldc
);
int dsyrk_(
    const char *uplo, const char *trans, const int *n, const int *k, const double *alpha, 
    const double *a, const int *lda, const double *beta, double *c, const int *ldc
);

// ========== LAPACK ========== //

void spotrf_(const char *uplo, const int *n, float  *a, const int *lda, int *info);
void dpotrf_(const char *uplo, const int *n, double *a, const int *lda, int *info);

void spotrs_(
    const char *uplo, const int *n, const int *nrhs, float  *a, const int *lda, 
    float  *b, const int *ldb, int *info
);
void dpotrs_(
    const char *uplo, const int *n, const int *nrhs, double *a, const int *lda, 
    double *b, const int *ldb, int *info
);

void sgesv_(
    const int *n, const int *nrhs, float  *a, const int *lda, int *ipiv, 
    float  *b, const int *ldb, int *info
);
void dgesv_(
    const int *n, const int *nrhs, double *a, const int *lda, int *ipiv, 
    double *b, const int *ldb, int *info
);

void sposv_(
    const char *uplo, const int *n, const int *nrhs, float  *a, const int *lda, 
    float  *b, const int *ldb, int *info
);
void dposv_(
    const char *uplo, const int *n, const int *nrhs, double *a, const int *lda, 
    double *b, const int *ldb, int *info
);

void ssyev_(
    const char *jobz, const char *uplo, const int *n, float  *a, const int *lda,
    float  *w, float  *work, const int *lwork, int *info
);
void dsyev_(
    const char *jobz, const char *uplo, const int *n, double *a, const int *lda,
    double *w, double *work, const int *lwork, int *info
);

void ssyevd_(
    const char *jobz, const char *uplo, const int *n, float  *a, const int *lda,
    float  *w, float  *work, const int *lwork, int *iwork, const int *liwork, int *info
);
void dsyevd_(
    const char *jobz, const char *uplo, const int *n, double *a, const int *lda,
    double *w, double *work, const int *lwork, int *iwork, const int *liwork, int *info
);

void sgesvd_(
    const char *jobu, const char *jobvt, const int *m, const int *n, 
    float  *a, const int *lda, float  *s, float  *u, const int *ldu, 
    float  *vt, const int *ldvt, float  *work, const int *lwork, int *info
);
void dgesvd_(
    const char *jobu, const char *jobvt, const int *m, const int *n, 
    double *a, const int *lda, double *s, double *u, const int *ldu, 
    double *vt, const int *ldvt, double *work, const int *lwork, int *info
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

#endif  // __CPU_LINALG_LP64_H__
