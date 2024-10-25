#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <bits/stdc++.h>
#include <omp.h>

#include "solvers/csr_mat.h"
#include "cpu_linalg.hpp"
#include "utils.h"

typedef std::pair<int, int> int_pair;

template <typename VT>
int test_function(const int argc, const char **argv)
{
    const int is_float  = std::is_same<VT, float>::value;
    const int is_double = std::is_same<VT, double>::value;

    if (is_double) printf("========== Test double precision ==========\n");
    if (is_float)  printf("========== Test single precision ==========\n");

    VT eps = is_double ? 1e-12 : 1e-6;
    val_type_t val_type = is_double ? VAL_TYPE_DOUBLE : VAL_TYPE_FLOAT;
    utils_dtype_t utils_dtype = is_double ? UTILS_DTYPE_FP64 : UTILS_DTYPE_FP32;

    int m     = atoi(argv[1]);
    int nnz   = atoi(argv[2]);
    int nvec  = atoi(argv[3]);
    int ntest = 5;
    if (argc >= 5) ntest = atoi(argv[4]);
    if (nnz < m)
    {
        nnz = 10 * m;
        printf("nnz must be >= m (%d), reset nnz to %d\n", m, nnz);
    }

    // Generate a random sparse lower triangular matrix
    printf("Generating a random sparse triangular matrix, size = %d, nnz = %d\n", m, nnz);
    size_t nnz_offdiag = (size_t) nnz - (size_t) m;
    int *row = (int *) malloc(sizeof(int) * nnz);
    int *col = (int *) malloc(sizeof(int) * nnz);
    VT  *val = (VT  *) malloc(sizeof(VT)  * nnz);
    VT diag_shift = (VT) nnz / (VT) m;
    srand(19241112);
    for (int i = 0; i < m; i++)
    {
        row[i] = i;
        col[i] = i;
        val[i] = ((VT) rand() / (VT) RAND_MAX) + diag_shift;
    }
    std::set<int_pair> nnz_set;
    while (nnz_set.size() < nnz_offdiag)
    {
        int i = rand() % m;
        int j = rand() % m;
        if (j < i) nnz_set.insert(int_pair(i, j));
    }
    for (int i = m; i < nnz; i++)
    {
        int_pair ij = *nnz_set.begin();
        nnz_set.erase(nnz_set.begin());
        row[i] = ij.first;
        col[i] = ij.second;
        val[i] = 2.0 * ((VT) rand() / (VT) RAND_MAX) - 1.0;
    }
    printf("Sparse triangular matrix generated\n");

    // Create lower and upper triangular matrices
    csr_mat_p L = NULL, U = NULL;
    coo_to_csr(val_type, m, m, nnz, row, col, val, &L);
    coo_to_csr(val_type, m, m, nnz, col, row, val, &U);

    // Create input and output matrices
    VT *X  = (VT *) malloc(sizeof(VT) * m * nvec);
    VT *LX = (VT *) malloc(sizeof(VT) * m * nvec);
    VT *UX = (VT *) malloc(sizeof(VT) * m * nvec);
    VT *XL = (VT *) malloc(sizeof(VT) * m * nvec);
    VT *XU = (VT *) malloc(sizeof(VT) * m * nvec);
    for (int i = 0; i < m * nvec; i++) X[i] = 2.0 * ((VT) rand() / (VT) RAND_MAX) - 1.0;

    double st, et;

    printf("SpMM L * X runtime (s)\n");
    for (int i = 0; i < ntest; i++)
    {
        st = omp_get_wtime();
        csr_spmm(L, nvec, X, m, LX, m);
        et = omp_get_wtime();
        printf("%.3f\n", et - st);
    }

    printf("SpMM U * X runtime (s)\n");
    for (int i = 0; i < ntest; i++)
    {
        st = omp_get_wtime();
        csr_spmm(U, nvec, X, m, UX, m);
        et = omp_get_wtime();
        printf("%.3f\n", et - st);
    }

    printf("SpTRSM L * X = B runtime (s)\n");
    for (int i = 0; i < ntest; i++)
    {
        st = omp_get_wtime();
        csr_trsm(lower, L, nvec, LX, m, XL, m);
        et = omp_get_wtime();
        printf("%.3f\n", et - st);
    }

    printf("SpTRSM U * X = B runtime (s)\n");
    for (int i = 0; i < ntest; i++)
    {
        st = omp_get_wtime();
        csr_trsm(upper, U, nvec, UX, m, XU, m);
        et = omp_get_wtime();
        printf("%.3f\n", et - st);
    }

    VT X_fnorm = 0.0, RL_fnorm = 0.0, RU_fnorm = 0.0;
    calc_err_2norm(utils_dtype, m * nvec, X, XL, &X_fnorm, &RL_fnorm);
    calc_err_2norm(utils_dtype, m * nvec, X, XU, &X_fnorm, &RU_fnorm);
    VT L_relerr = RL_fnorm / X_fnorm;
    VT U_relerr = RU_fnorm / X_fnorm;
    printf("SpTRSM for L residual F-norm = %e\n", L_relerr);
    printf("SpTRSM for U residual F-norm = %e\n", U_relerr);
    int test_passed = (L_relerr < eps && U_relerr < eps) ? 1 : 0;
    if (test_passed) printf("Test passed\n\n");
    else printf("Test failed\n\n");

    csr_mat_free(&L);
    csr_mat_free(&U);
    free(row);
    free(col);
    free(val);
    free(X);
    free(LX);
    free(UX);
    free(XL);
    free(XU);
    return test_passed;
}

int main(const int argc, const char **argv)
{
    if (argc < 4)
    {
        printf("Usage: %s m nnz nvec ntest\n", argv[0]);
        printf("m    : Size of the sparse lower / upper trianglar matrix\n");
        printf("nnz  : Number of non-zeros\n");
        printf("nvec : Number of vectors in the right-hand-side\n");
        return 1;
    }

    int fp32_passed = test_function<float>(argc, argv);
    int fp64_passed = test_function<double>(argc, argv);
    int test_passed = fp32_passed && fp64_passed;
    printf("Are all tests passed? %s\n\n", test_passed ? "YES" : "NO");
    
    return (1 - test_passed);
}
