#ifndef __CPU_LINALG_ILP64_HPP__
#define __CPU_LINALG_ILP64_HPP__

#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include "cpu_linalg_ilp64.h"

#ifdef __cplusplus

// ========== BLAS ========== //

static float  xnrm2_(const int *n, const float  *x, const int *incx) 
{
    int64_t n64 = *n, incx64 = *incx;
    return snrm2_64_(&n64, x, &incx64); 
}
static double xnrm2_(const int *n, const double *x, const int *incx) 
{
    int64_t n64 = *n, incx64 = *incx;
    return dnrm2_64_(&n64, x, &incx64); 
}

static float  xdot_(const int *n, const float  *x, const int *incx, const float  *y, const int *incy) 
{
    int64_t n64 = *n, incx64 = *incx, incy64 = *incy;
    return sdot_64_(&n64, x, &incx64, y, &incy64);
}
static double xdot_(const int *n, const double *x, const int *incx, const double *y, const int *incy) 
{
    int64_t n64 = *n, incx64 = *incx, incy64 = *incy;
    return ddot_64_(&n64, x, &incx64, y, &incy64);
}

static int xgemv_(
    const char *trans, const int *m, const int *n,
    const float *alpha, const float *a, const int *lda, const float *x, const int *incx,
    const float *beta, float *y, const int *incy
)
{
    int64_t m64 = *m, n64 = *n, lda64 = *lda, incx64 = *incx, incy64 = *incy;
    return sgemv_64_(trans, &m64, &n64, alpha, a, &lda64, x, &incx64, beta, y, &incy64);
}
static int xgemv_(
    const char *trans, const int *m, const int *n,
    const double *alpha, const double *a, const int *lda, const double *x, const int *incx,
    const double *beta, double *y, const int *incy
)
{
    int64_t m64 = *m, n64 = *n, lda64 = *lda, incx64 = *incx, incy64 = *incy;
    return dgemv_64_(trans, &m64, &n64, alpha, a, &lda64, x, &incx64, beta, y, &incy64);
}

static int xgemm_(
    const char *transa, const char *transb, const int *m, const int *n, const int *k,
    const float *alpha, const float *a, const int *lda, const float *b, const int *ldb,
    const float *beta, float *c, const int *ldc
)
{
    int64_t m64 = *m, n64 = *n, k64 = *k, lda64 = *lda, ldb64 = *ldb, ldc64 = *ldc;
    return sgemm_64_(transa, transb, &m64, &n64, &k64, alpha, a, &lda64, b, &ldb64, beta, c, &ldc64);
}
static int xgemm_(
    const char *transa, const char *transb, const int *m, const int *n, const int *k,
    const double *alpha, const double *a, const int *lda, const double *b, const int *ldb,
    const double *beta, double *c, const int *ldc
)
{
    int64_t m64 = *m, n64 = *n, k64 = *k, lda64 = *lda, ldb64 = *ldb, ldc64 = *ldc;
    return dgemm_64_(transa, transb, &m64, &n64, &k64, alpha, a, &lda64, b, &ldb64, beta, c, &ldc64);
}

static int xtrsm_(
    const char *side, const char *uplo, const char *transa, const char *diag, const int *m, const int *n, 
    const float  *alpha, const float  *a, const int *lda, float  *b, const int *ldb
)
{
    int64_t m64 = *m, n64 = *n, lda64 = *lda, ldb64 = *ldb;
    return strsm_64_(side, uplo, transa, diag, &m64, &n64, alpha, a, &lda64, b, &ldb64);
}
static int xtrsm_(
    const char *side, const char *uplo, const char *transa, const char *diag, const int *m, const int *n, 
    const double *alpha, const double *a, const int *lda, double *b, const int *ldb
)
{
    int64_t m64 = *m, n64 = *n, lda64 = *lda, ldb64 = *ldb;
    return dtrsm_64_(side, uplo, transa, diag, &m64, &n64, alpha, a, &lda64, b, &ldb64);
}

static int xsyrk_(
    const char *uplo, const char *trans, const int *n, const int *k, const float  *alpha, 
    const float  *a, const int *lda, const float  *beta, float  *c, const int *ldc
)
{
    int64_t n64 = *n, k64 = *k, lda64 = *lda, ldc64 = *ldc;
    return ssyrk_64_(uplo, trans, &n64, &k64, alpha, a, &lda64, beta, c, &ldc64);
}
static int xsyrk_(
    const char *uplo, const char *trans, const int *n, const int *k, const double *alpha, 
    const double *a, const int *lda, const double *beta, double *c, const int *ldc
)
{
    int64_t n64 = *n, k64 = *k, lda64 = *lda, ldc64 = *ldc;
    return dsyrk_64_(uplo, trans, &n64, &k64, alpha, a, &lda64, beta, c, &ldc64);
}

// ========== LAPACK ========== //

static void xpotrf_(const char *uplo, const int *n, float  *a, const int *lda, int *info)
{
    int64_t n64 = *n, lda64 = *lda, info64 = 0;
    spotrf_64_(uplo, &n64, a, &lda64, &info64);
    *info = (int) info64;
}
static void xpotrf_(const char *uplo, const int *n, double *a, const int *lda, int *info)
{
    int64_t n64 = *n, lda64 = *lda, info64 = 0;
    dpotrf_64_(uplo, &n64, a, &lda64, &info64);
    *info = (int) info64;
}

static void xpotrs_(
    const char *uplo, const int *n, const int *nrhs, float  *a, const int *lda, 
    float  *b, const int *ldb, int *info
)
{
    int64_t n64 = *n, nrhs64 = *nrhs, lda64 = *lda, ldb64 = *ldb, info64 = 0;
    spotrs_64_(uplo, &n64, &nrhs64, a, &lda64, b, &ldb64, &info64);
    *info = (int) info64;
}
static void xpotrs_(
    const char *uplo, const int *n, const int *nrhs, double *a, const int *lda, 
    double *b, const int *ldb, int *info
)
{
    int64_t n64 = *n, nrhs64 = *nrhs, lda64 = *lda, ldb64 = *ldb, info64 = 0;
    dpotrs_64_(uplo, &n64, &nrhs64, a, &lda64, b, &ldb64, &info64);
    *info = (int) info64;
}

static void xgesv_(
    const int *n, const int *nrhs, float  *a, const int *lda, int *ipiv, 
    float  *b, const int *ldb, int *info
)
{
    int64_t n64 = *n, nrhs64 = *nrhs, lda64 = *lda, ldb64 = *ldb, info64 = 0;
    int64_t *ipiv64 = (int64_t *) malloc(sizeof(int64_t) * (size_t) n64);
    sgesv_64_(&n64, &nrhs64, a, &lda64, ipiv64, b, &ldb64, &info64);
    for (int i = 0; i < *n; i++) ipiv[i] = (int) ipiv64[i];
    *info = (int) info64;
    free(ipiv64);
}
static void xgesv_(
    const int *n, const int *nrhs, double *a, const int *lda, int *ipiv, 
    double *b, const int *ldb, int *info
)
{
    int64_t n64 = *n, nrhs64 = *nrhs, lda64 = *lda, ldb64 = *ldb, info64 = 0;
    int64_t *ipiv64 = (int64_t *) malloc(sizeof(int64_t) * (size_t) n64);
    dgesv_64_(&n64, &nrhs64, a, &lda64, ipiv64, b, &ldb64, &info64);
    for (int i = 0; i < *n; i++) ipiv[i] = (int) ipiv64[i];
    *info = (int) info64;
    free(ipiv64);
}

static void xposv_(
    const char *uplo, const int *n, const int *nrhs, float  *a, const int *lda, 
    float  *b, const int *ldb, int *info
)
{
    int64_t n64 = *n, nrhs64 = *nrhs, lda64 = *lda, ldb64 = *ldb, info64 = 0;
    sposv_64_(uplo, &n64, &nrhs64, a, &lda64, b, &ldb64, &info64);
    *info = (int) info64;
}
static void xposv_(
    const char *uplo, const int *n, const int *nrhs, double *a, const int *lda, 
    double *b, const int *ldb, int *info
)
{
    int64_t n64 = *n, nrhs64 = *nrhs, lda64 = *lda, ldb64 = *ldb, info64 = 0;
    dposv_64_(uplo, &n64, &nrhs64, a, &lda64, b, &ldb64, &info64);
    *info = (int) info64;
}

static void xsyev_(
    const char *jobz, const char *uplo, const int *n, float  *a, const int *lda,
    float  *w, float  *work, const int *lwork, int *info
)
{
    int64_t n64 = *n, lda64 = *lda, lwork64 = *lwork, info64 = 0;
    ssyev_64_(jobz, uplo, &n64, a, &lda64, w, work, &lwork64, &info64);
    *info = (int) info64;
}
static void xsyev_(
    const char *jobz, const char *uplo, const int *n, double *a, const int *lda,
    double *w, double *work, const int *lwork, int *info
)
{
    int64_t n64 = *n, lda64 = *lda, lwork64 = *lwork, info64 = 0;
    dsyev_64_(jobz, uplo, &n64, a, &lda64, w, work, &lwork64, &info64);
    *info = (int) info64;
}

static void xsyevd_(
    const char *jobz, const char *uplo, const int *n, float  *a, const int *lda,
    float  *w, float  *work, const int *lwork, int *iwork, const int *liwork, int *info
)
{
    int64_t n64 = *n, lda64 = *lda, lwork64 = *lwork, liwork64 = *liwork, info64 = 0;
    if (liwork64 == -1)
    {
        // Query workspace size, optimal lwork is stored at work[0], optimal liwork is stored at iwork64 -> iwork[0]
        int64_t iwork64 = 0;
        ssyevd_64_(jobz, uplo, &n64, a, &lda64, w, work, &lwork64, &iwork64, &liwork64, &info64);
        *iwork = (int) iwork64;
    }
    if (liwork64 > 0)
    {
        // Normal execution, need to allocate iwork64
        int64_t *iwork64 = (int64_t *) malloc(sizeof(int64_t) * (size_t) liwork64);
        ssyevd_64_(jobz, uplo, &n64, a, &lda64, w, work, &lwork64, iwork64, &liwork64, &info64);
        free(iwork64);
    }
    *info = (int) info64;
}
static void xsyevd_(
    const char *jobz, const char *uplo, const int *n, double *a, const int *lda,
    double *w, double *work, const int *lwork, int *iwork, const int *liwork, int *info
)
{
    int64_t n64 = *n, lda64 = *lda, lwork64 = *lwork, liwork64 = *liwork, info64 = 0;
    if (liwork64 == -1)
    {
        // Query workspace size, optimal lwork is stored at work[0], optimal liwork is stored at iwork64 -> iwork[0]
        int64_t iwork64 = 0;
        dsyevd_64_(jobz, uplo, &n64, a, &lda64, w, work, &lwork64, &iwork64, &liwork64, &info64);
        *iwork = (int) iwork64;
    }
    if (liwork64 > 0)
    {
        // Normal execution, need to allocate iwork64
        int64_t *iwork64 = (int64_t *) malloc(sizeof(int64_t) * (size_t) liwork64);
        dsyevd_64_(jobz, uplo, &n64, a, &lda64, w, work, &lwork64, iwork64, &liwork64, &info64);
        free(iwork64);
    }
    *info = (int) info64;
}

static void xgesvd_(
    const char *jobu, const char *jobvt, const int *m, const int *n, 
    float  *a, const int *lda, float  *s, float  *u, const int *ldu, 
    float  *vt, const int *ldvt, float  *work, const int *lwork, int *info
)
{
    int64_t m64 = *m, n64 = *n, lda64 = *lda, ldu64 = *ldu;
    int64_t ldvt64 = *ldvt, lwork64 = *lwork, info64 = 0;
    sgesvd_64_(jobu, jobvt, &m64, &n64, a, &lda64, s, u, &ldu64, vt, &ldvt64, work, &lwork64, &info64);
    *info = (int) info64;
}
static void xgesvd_(
    const char *jobu, const char *jobvt, const int *m, const int *n, 
    double *a, const int *lda, double *s, double *u, const int *ldu, 
    double *vt, const int *ldvt, double *work, const int *lwork, int *info
)
{
    int64_t m64 = *m, n64 = *n, lda64 = *lda, ldu64 = *ldu;
    int64_t ldvt64 = *ldvt, lwork64 = *lwork, info64 = 0;
    dgesvd_64_(jobu, jobvt, &m64, &n64, a, &lda64, s, u, &ldu64, vt, &ldvt64, work, &lwork64, &info64);
    *info = (int) info64;
}

#endif  // __cplusplus

#endif  // __CPU_LINALG_ILP64_HPP__