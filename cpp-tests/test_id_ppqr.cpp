#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <type_traits>

#include "common.h"
#include "cpu_linalg.hpp"
#include "h2mat/id_ppqr.h"
#include "utils.h"

template <typename VT>
int test_function(const int argc, const char **argv)
{
    const int is_float  = std::is_same<VT, float>::value;
    const int is_double = std::is_same<VT, double>::value;

    if (is_double) printf("========== Test double precision ==========\n");
    if (is_float)  printf("========== Test single precision ==========\n");

    val_type_t val_type = is_double ? VAL_TYPE_DOUBLE : VAL_TYPE_FLOAT;

    int m = atoi(argv[1]);
    int n = atoi(argv[2]);
    VT rel_tol = (VT) atof(argv[3]);

    if (is_float && rel_tol < 1e-5f)
    {
        printf("Single precision test cannot use rel_tol < 1e-5. Reset rel_tol to 1e-5.\n");
        rel_tol = 1e-5f;
    }

    VT *A  = (VT *) malloc(sizeof(VT) * m * n);
    VT *A0 = (VT *) malloc(sizeof(VT) * m * n);
    VT *c0 = (VT *) malloc(sizeof(VT) * n * 2);
    VT *c1 = (VT *) malloc(sizeof(VT) * m * 2);
    for (int i = 0; i < n * 2; i++) c0[i] = (VT) rand() / (VT) RAND_MAX;
    for (int i = 0; i < m * 2; i++) c1[i] = (VT) rand() / (VT) RAND_MAX + 0.5;
    VT A_fnorm = 0.0;
    srand(19241112);
    for (int j = 0; j < n; j++)
    {
        VT x0 = c0[2 * j];
        VT y0 = c0[2 * j + 1];
        for (int i = 0; i < m; i++)
        {
            VT dx = c1[2 * i] - x0;
            VT dy = c1[2 * i + 1] - y0;
            VT d2 = dx * dx + dy * dy;
            A[i + j * m] = 1.0 / std::sqrt(d2);
            A_fnorm += A[i + j * m] * A[i + j * m];
        }
    }
    memcpy(A0, A, sizeof(VT) * m * n);
    A_fnorm = std::sqrt(A_fnorm);

    int rank = 0, max_rank = 0, n_thread = 0;
    VT *V = NULL;
    int *col_idx = NULL;
    id_ppqr(
        m, n, val_type, A, m, 
        max_rank, (const void *) &rel_tol, n_thread,
        &rank, &col_idx, (void **) &V, NULL, NULL
    );
    printf("rank = %d\n", rank);

    VT *AJ = (VT *) malloc(sizeof(VT) * m * n);
    for (int i = 0; i < rank; i++)
        memcpy(AJ + i * m, A0 + col_idx[i] * m, sizeof(VT) * m);
    const char *notrans = "N";
    VT v_one = 1.0, v_neg_one = -1.0;
    xgemm_(notrans, notrans, &m, &n, &rank, &v_one, AJ, &m, V, &rank, &v_neg_one, A0, &m);
    VT diff_fnorm = 0.0;
    for (int i = 0; i < m * n; i++) diff_fnorm += A0[i] * A0[i];
    diff_fnorm = std::sqrt(diff_fnorm);
    VT relerr = diff_fnorm / A_fnorm;
    printf("||A - A(:, J) * V||_F / ||A||_F = %e\n", relerr);

    VT prefac = std::log10(static_cast<VT>((m > n) ? m : n));  // How to choose this?
    int test_passed = (relerr < rel_tol * prefac) ? 1 : 0;
    if (test_passed) printf("Test passed\n\n");
    else printf("Test failed\n\n");

    free(A);
    free(A0);
    free(AJ);
    free(V);
    free(col_idx);
    free(c0);
    free(c1);
    return test_passed;
}

int main(const int argc, const char **argv)
{
    if (argc < 4)
    {
        fprintf(stderr, "Usage: %s m n reltol\n", argv[0]);
        fprintf(stderr, "  m, n   : Size of the matrix\n");
        fprintf(stderr, "  reltol : Relative tolerance of the ID\n");
        return 255;
    }

    int fp32_passed = test_function<float>(argc, argv);
    int fp64_passed = test_function<double>(argc, argv);
    int test_passed = fp32_passed && fp64_passed;
    printf("Are all tests passed? %s\n\n", test_passed ? "YES" : "NO");
    
    return (1 - test_passed);
}
