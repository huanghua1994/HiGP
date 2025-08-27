#ifndef __CPU_LINALG_LP64_HPP__
#define __CPU_LINALG_LP64_HPP__

#include "cpu_linalg_lp64.h"

#ifdef __cplusplus

// ========== BLAS ========== //

static float  xnrm2_(const int *n, const float  *x, const int *incx) 
{
    return snrm2_(n, x, incx); 
}
static double xnrm2_(const int *n, const double *x, const int *incx) 
{
    return dnrm2_(n, x, incx); 
}

static float  xdot_(const int *n, const float  *x, const int *incx, const float  *y, const int *incy) 
{
    return sdot_(n, x, incx, y, incy); 
}
static double xdot_(const int *n, const double *x, const int *incx, const double *y, const int *incy) 
{
    return ddot_(n, x, incx, y, incy); 
}

static int xgemv_(
    const char *trans, const int *m, const int *n,
    const float *alpha, const float *a, const int *lda, const float *x, const int *incx,
    const float *beta, float *y, const int *incy
)
{
#ifdef USE_ACCELERATE_LP64
    sgemv_(trans, m, n, alpha, a, lda, x, incx, beta, y, incy);
    return 0;
#else
    return sgemv_(trans, m, n, alpha, a, lda, x, incx, beta, y, incy);
#endif
}
static int xgemv_(
    const char *trans, const int *m, const int *n,
    const double *alpha, const double *a, const int *lda, const double *x, const int *incx,
    const double *beta, double *y, const int *incy
)
{
#ifdef USE_ACCELERATE_LP64
    dgemv_(trans, m, n, alpha, a, lda, x, incx, beta, y, incy);
    return 0;
#else
    return dgemv_(trans, m, n, alpha, a, lda, x, incx, beta, y, incy);
#endif
}

static int xgemm_(
    const char *transa, const char *transb, const int *m, const int *n, const int *k,
    const float *alpha, const float *a, const int *lda, const float *b, const int *ldb,
    const float *beta, float *c, const int *ldc
)
{
#ifdef USE_ACCELERATE_LP64
    sgemm_(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
    return 0;
#else
    return sgemm_(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
#endif
}
static int xgemm_(
    const char *transa, const char *transb, const int *m, const int *n, const int *k,
    const double *alpha, const double *a, const int *lda, const double *b, const int *ldb,
    const double *beta, double *c, const int *ldc
)
{
#ifdef USE_ACCELERATE_LP64
    dgemm_(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
    return 0;
#else
    return dgemm_(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
#endif
}

static int xtrsm_(
    const char *side, const char *uplo, const char *transa, const char *diag, const int *m, const int *n, 
    const float  *alpha, const float  *a, const int *lda, float  *b, const int *ldb
)
{
#ifdef USE_ACCELERATE_LP64
    strsm_(side, uplo, transa, diag, m, n, alpha, a, lda, b, ldb);
    return 0;
#else
    return strsm_(side, uplo, transa, diag, m, n, alpha, a, lda, b, ldb);
#endif
}
static int xtrsm_(
    const char *side, const char *uplo, const char *transa, const char *diag, const int *m, const int *n, 
    const double *alpha, const double *a, const int *lda, double *b, const int *ldb
)
{
#ifdef USE_ACCELERATE_LP64
    dtrsm_(side, uplo, transa, diag, m, n, alpha, a, lda, b, ldb);
    return 0;
#else
    return dtrsm_(side, uplo, transa, diag, m, n, alpha, a, lda, b, ldb);
#endif
}

static int xsyrk_(
    const char *uplo, const char *trans, const int *n, const int *k, const float  *alpha, 
    const float  *a, const int *lda, const float  *beta, float  *c, const int *ldc
)
{
#ifdef USE_ACCELERATE_LP64
    ssyrk_(uplo, trans, n, k, alpha, a, lda, beta, c, ldc);
    return 0;
#else
    return ssyrk_(uplo, trans, n, k, alpha, a, lda, beta, c, ldc);
#endif
}
static int xsyrk_(
    const char *uplo, const char *trans, const int *n, const int *k, const double *alpha, 
    const double *a, const int *lda, const double *beta, double *c, const int *ldc
)
{
#ifdef USE_ACCELERATE_LP64
    dsyrk_(uplo, trans, n, k, alpha, a, lda, beta, c, ldc);
    return 0;
#else
    return dsyrk_(uplo, trans, n, k, alpha, a, lda, beta, c, ldc);
#endif
}

// ========== LAPACK ========== //

static void xpotrf_(const char *uplo, const int *n, float  *a, const int *lda, int *info)
{
    spotrf_(uplo, n, a, lda, info);
}
static void xpotrf_(const char *uplo, const int *n, double *a, const int *lda, int *info)
{
    dpotrf_(uplo, n, a, lda, info);
}

static void xpotrs_(
    const char *uplo, const int *n, const int *nrhs, float  *a, const int *lda, 
    float  *b, const int *ldb, int *info
)
{
    spotrs_(uplo, n, nrhs, a, lda, b, ldb, info);
}
static void xpotrs_(
    const char *uplo, const int *n, const int *nrhs, double *a, const int *lda, 
    double *b, const int *ldb, int *info
)
{
    dpotrs_(uplo, n, nrhs, a, lda, b, ldb, info);
}

static void xgesv_(
    const int *n, const int *nrhs, float  *a, const int *lda, int *ipiv, 
    float  *b, const int *ldb, int *info
)
{
    sgesv_(n, nrhs, a, lda, ipiv, b, ldb, info);
}
static void xgesv_(
    const int *n, const int *nrhs, double *a, const int *lda, int *ipiv, 
    double *b, const int *ldb, int *info
)
{
    dgesv_(n, nrhs, a, lda, ipiv, b, ldb, info);
}

static void xposv_(
    const char *uplo, const int *n, const int *nrhs, float  *a, const int *lda, 
    float  *b, const int *ldb, int *info
)
{
    sposv_(uplo, n, nrhs, a, lda, b, ldb, info);
}
static void xposv_(
    const char *uplo, const int *n, const int *nrhs, double *a, const int *lda, 
    double *b, const int *ldb, int *info
)
{
    dposv_(uplo, n, nrhs, a, lda, b, ldb, info);
}

static void xsyev_(
    const char *jobz, const char *uplo, const int *n, float  *a, const int *lda,
    float  *w, float  *work, const int *lwork, int *info
)
{
    ssyev_(jobz, uplo, n, a, lda, w, work, lwork, info);
}
static void xsyev_(
    const char *jobz, const char *uplo, const int *n, double *a, const int *lda,
    double *w, double *work, const int *lwork, int *info
)
{
    dsyev_(jobz, uplo, n, a, lda, w, work, lwork, info);
}

static void xsyevd_(
    const char *jobz, const char *uplo, const int *n, float  *a, const int *lda,
    float  *w, float  *work, const int *lwork, int *iwork, const int *liwork, int *info
)
{
    ssyevd_(jobz, uplo, n, a, lda, w, work, lwork, iwork, liwork, info);
}
static void xsyevd_(
    const char *jobz, const char *uplo, const int *n, double *a, const int *lda,
    double *w, double *work, const int *lwork, int *iwork, const int *liwork, int *info
)
{
    dsyevd_(jobz, uplo, n, a, lda, w, work, lwork, iwork, liwork, info);
}

static void xgesvd_(
    const char *jobu, const char *jobvt, const int *m, const int *n, 
    float  *a, const int *lda, float  *s, float  *u, const int *ldu, 
    float  *vt, const int *ldvt, float  *work, const int *lwork, int *info
)
{
    sgesvd_(jobu, jobvt, m, n, a, lda, s, u, ldu, vt, ldvt, work, lwork, info);
}
static void xgesvd_(
    const char *jobu, const char *jobvt, const int *m, const int *n, 
    double *a, const int *lda, double *s, double *u, const int *ldu, 
    double *vt, const int *ldvt, double *work, const int *lwork, int *info
)
{
    dgesvd_(jobu, jobvt, m, n, a, lda, s, u, ldu, vt, ldvt, work, lwork, info);
}

#endif  // __cplusplus

#endif  // __CPU_LINALG_LP64_HPP__
