#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <type_traits>

#include "dense_kernel_matrix.h"
#include "solvers/solvers.h"
#include "utils.h"

template <typename VT>
int test_function(const int argc, const char **argv)
{
    const int is_float  = std::is_same<VT, float>::value;
    const int is_double = std::is_same<VT, double>::value;

    if (is_double) printf("========== Test double precision ==========\n");
    if (is_float)  printf("========== Test single precision ==========\n");

    VT reltol = is_double ? 1e-9 : 1e-4;
    val_type_t val_type = is_double ? VAL_TYPE_DOUBLE : VAL_TYPE_FLOAT;
    utils_dtype_t utils_dtype = is_double ? UTILS_DTYPE_FP64 : UTILS_DTYPE_FP32;

    int m      = atoi(argv[1]);
    int dim    = atoi(argv[2]);
    int nvec   = atoi(argv[3]);
    int n_iter = atoi(argv[4]);

    int test_passed = 1;
    double st, et;

    int T_size = nvec * n_iter * n_iter;
    size_t bd_size = 3;         // l, f, s
    bd_size += m * dim;         // coord
    bd_size += m;               // dnoise
    bd_size += m * nvec;        // B
    bd_size += m * nvec;        // X of mpcg
    bd_size += T_size;          // T of mpcg
    bd_size += m * nvec;        // X of mfom
    bd_size += T_size;          // T of mfom
    bd_size *= sizeof(double);  // Input binary are all double precision values
    double *bd_data = (double *) malloc(bd_size);
    read_binary_file(argv[5], bd_data, bd_size);

    double *bd_ptr = bd_data;
    VT l = static_cast<VT>(bd_ptr[0]);
    VT f = static_cast<VT>(bd_ptr[1]);
    VT s = static_cast<VT>(bd_ptr[2]);
    VT param[4] = {(VT) dim, l, f, s};
    bd_ptr += 3;

    VT *coord = (VT *) malloc(sizeof(VT) * m * dim);
    for (int i = 0; i < m * dim; i++) coord[i] = static_cast<VT>(bd_ptr[i]);
    bd_ptr += m * dim;

    VT *dnoise = (VT *) malloc(sizeof(VT) * m);
    for (int i = 0; i < m; i++) dnoise[i] = static_cast<VT>(bd_ptr[i]);
    bd_ptr += m;

    int krnl_id = KERNEL_ID_GAUSSIAN;
    printf("Use kernel: %s\n", krnl_id == 1 ? "Gaussian" : "Matern32");
    printf("param: l = %.4f, f = %.4f, s = %.6e\n", l, f, s);

    dense_krnl_mat_p dkmat;
    dense_krnl_mat_init(
        m, m, coord, m, m, coord, 
        param, dnoise, krnl_id, val_type, &dkmat
    );
    st = get_wtime_sec();
    dense_krnl_mat_populate(dkmat);
    et = get_wtime_sec();
    printf("dense_krnl_mat_populate() done, time = %.3f s\n", et - st);
    matmul_fptr A_mm = (matmul_fptr) dense_krnl_mat_krnl_matmul;

    VT *B     = (VT *) malloc(sizeof(VT) * m * nvec);
    VT *X     = (VT *) malloc(sizeof(VT) * m * nvec);
    VT *T     = (VT *) malloc(sizeof(VT) * T_size);
    VT *ref_X = (VT *) malloc(sizeof(VT) * m * nvec);
    VT *ref_T = (VT *) malloc(sizeof(VT) * T_size);

    for (int i = 0; i < m * nvec; i++) B[i] = static_cast<VT>(bd_ptr[i]);
    bd_ptr += m * nvec;

    // Test mpcg
    for (int i = 0; i < m * nvec; i++) ref_X[i] = static_cast<VT>(bd_ptr[i]);
    bd_ptr += m * nvec;
    for (int i = 0; i < T_size; i++) ref_T[i] = static_cast<VT>(bd_ptr[i]);
    bd_ptr += T_size;

    memset(X, 0, sizeof(VT) * m * nvec);
    memset(T, 0, sizeof(VT) * T_size);
    st = get_wtime_sec();
    mpcg(
        m, nvec, n_iter, val_type,
        A_mm, dkmat, NULL, NULL, 
        B, m, X, m, T
    );
    et = get_wtime_sec();
    printf("mpcg() finished, time = %.3f s\n", et - st);

    VT ref_X_fnorm = 0, ref_T_fnorm = 0;
    VT X_relerr = 0, T_relerr = 0;

    // Single precision results might not be accurate enough compared to
    // double precision reference results
    calc_err_2norm(utils_dtype, m * nvec, ref_X, X, &ref_X_fnorm, &X_relerr);
    X_relerr /= ref_X_fnorm;
    if (is_float)
    {
        printf("mpcg() X relative error = %.6e\n", X_relerr);
    } else {
        printf("mpcg() X relative error = %.6e, %s\n", X_relerr, (X_relerr < reltol) ? "PASSED" : "FAILED");
        test_passed = test_passed && (X_relerr < reltol);
    }

    // Test mfom
    for (int i = 0; i < m * nvec; i++) ref_X[i] = static_cast<VT>(bd_ptr[i]);
    bd_ptr += m * nvec;
    for (int i = 0; i < T_size; i++) ref_T[i] = static_cast<VT>(bd_ptr[i]);
    bd_ptr += T_size;

    memset(X, 0, sizeof(VT) * m * nvec);
    memset(T, 0, sizeof(VT) * T_size);
    st = get_wtime_sec();
    mfom(
        m, nvec, n_iter, val_type,
        A_mm, dkmat, B, m, X, m, T
    );
    et = get_wtime_sec();
    printf("mfom() finished, time = %.3f s\n", et - st);

    calc_err_2norm(utils_dtype, m * nvec, ref_X, X, &ref_X_fnorm, &X_relerr);
    calc_err_2norm(utils_dtype, T_size,   ref_T, T, &ref_T_fnorm, &T_relerr);
    X_relerr /= ref_X_fnorm;
    T_relerr /= ref_T_fnorm;
    printf("mfom() X relative error = %.6e, %s\n", X_relerr, (X_relerr < reltol) ? "PASSED" : "FAILED");
    printf("mfom() T relative error = %.6e, %s\n", T_relerr, (T_relerr < reltol) ? "PASSED" : "FAILED");
    test_passed = test_passed && (X_relerr < reltol);
    test_passed = test_passed && (T_relerr < reltol);

    if (test_passed) printf("Test passed\n\n");
    else printf("Test failed\n\n");

    free(bd_data);
    free(coord);
    free(dnoise);
    free(B);
    free(X);
    free(T);
    free(ref_X);
    free(ref_T);
    return test_passed;
}

int main(const int argc, const char **argv)
{
    if (argc < 6)
    {
        printf("Usage: %s m dim nvec n_iter bd_file\n", argv[0]);
        printf("m       : Number of points \n");
        printf("dim     : Point dimension\n");
        printf("nvec    : Number of vectors in the right-hand-side\n");
        printf("krnl    : f^2 * K(X, X, l) + s * I\n");
        printf("bd_file : Binary file containing the input and reference output data, \n");
        printf("          generated using gen_tsolvers_test_bin.m\n");
        return 1;
    }

    int fp32_passed = test_function<float>(argc, argv);
    int fp64_passed = test_function<double>(argc, argv);
    int test_passed = fp32_passed && fp64_passed;
    printf("Are all tests passed? %s\n\n", test_passed ? "YES" : "NO");
    
    return (1 - test_passed);
}