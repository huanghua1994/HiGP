#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <type_traits>

#include "common.h"
#include "kernels/kernels.h"
#include "utils.h"

template <typename VT>
int test_function(const int argc, const char **argv)
{
    const int is_float  = std::is_same<VT, float>::value;
    const int is_double = std::is_same<VT, double>::value;

    if (is_double) printf("========== Test double precision ==========\n");
    if (is_float)  printf("========== Test single precision ==========\n");

    VT reltol = is_double ? 1e-14 : 1e-6;
    val_type_t val_type = is_double ? VAL_TYPE_DOUBLE : VAL_TYPE_FLOAT;
    utils_dtype_t utils_dtype = is_double ? UTILS_DTYPE_FP64 : UTILS_DTYPE_FP32;

    int n0  = atoi(argv[1]);
    int n1  = atoi(argv[2]);
    int dim = atoi(argv[3]);

    int test_passed = 1;
    double st, et;

    size_t bd_size = 1;         // l
    bd_size += n0 * dim;        // c0
    bd_size += n1 * dim;        // c1
    bd_size += n0 * n1;         // k_mat for pdist2 kernel
    bd_size += n0 * n1;         // k_mat for Gaussian kernel
    bd_size += n0 * n1;         // k_mat for Matern 3/2 kernel
    bd_size += n0 * n1;         // k_mat for Matern 5/2 kernel
    bd_size += n0 * n1;         // dl_mat for Gaussian kernel
    bd_size += n0 * n1;         // dl_mat for Matern 3/2 kernel
    bd_size += n0 * n1;         // dl_mat for Matern 5/2 kernel
    bd_size *= sizeof(double);  // Input binary are all double precision values
    double *bd_data = (double *) malloc(bd_size);
    read_binary_file(argv[4], bd_data, bd_size);

    double *bd_ptr = bd_data;
    VT l = static_cast<VT>(bd_ptr[0]);
    bd_ptr += 1;

    VT *c0 = (VT *) malloc(sizeof(VT) * n0 * dim);
    VT *c1 = (VT *) malloc(sizeof(VT) * n1 * dim);
    for (int i = 0; i < n0 * dim; i++) c0[i] = static_cast<VT>(bd_ptr[i]);
    bd_ptr += n0 * dim;
    for (int i = 0; i < n1 * dim; i++) c1[i] = static_cast<VT>(bd_ptr[i]);
    bd_ptr += n1 * dim;

    VT *k_mat      = (VT *) malloc(sizeof(VT) * n0 * n1);
    VT *dl_mat     = (VT *) malloc(sizeof(VT) * n0 * n1);
    VT *ref_k_mat  = (VT *) malloc(sizeof(VT) * n0 * n1);
    VT *ref_dl_mat = (VT *) malloc(sizeof(VT) * n0 * n1);

    VT param[2] = {(VT) dim, l};
    VT ref_k_mat_fnorm = 0, ref_dl_mat_fnorm = 0;
    VT k_mat_relerr = 0, dl_mat_relerr = 0;

    // Test pdist2 kernel
    for (int i = 0; i < n0 * n1; i++) ref_k_mat[i] = static_cast<VT>(bd_ptr[i]);
    bd_ptr += n0 * n1;
    st = get_wtime_sec();
    pdist2_krnl(n0, n0, c0, n1, n1, c1, param, n0, k_mat, val_type);
    et = get_wtime_sec();
    printf("pdist2_krnl() time: %.3f s\n", et - st);
    calc_err_2norm(utils_dtype, n0 * n1, ref_k_mat, k_mat, &ref_k_mat_fnorm, &k_mat_relerr);
    k_mat_relerr /= ref_k_mat_fnorm;
    printf("pdist2 kernel relerr: %.3e, %s\n", k_mat_relerr, (k_mat_relerr < reltol) ? "PASSED" : "FAILED");
    test_passed = test_passed && (k_mat_relerr < reltol);

    // Test Gaussian kernel and its derivative
    for (int i = 0; i < n0 * n1; i++) ref_k_mat[i] = static_cast<VT>(bd_ptr[i]);
    bd_ptr += n0 * n1;
    for (int i = 0; i < n0 * n1; i++) ref_dl_mat[i] = static_cast<VT>(bd_ptr[i]);
    bd_ptr += n0 * n1;
    st = get_wtime_sec();
    gaussian_krnl_grad(
        n0, n0, c0, n1, n1, c1,
        param, n0, val_type,
        1, k_mat, 1, dl_mat
    );
    et = get_wtime_sec();
    printf("gaussian_krnl_grad() time: %.3f s\n", et - st);
    calc_err_2norm(utils_dtype, n0 * n1, ref_k_mat,  k_mat,  &ref_k_mat_fnorm,  &k_mat_relerr);
    calc_err_2norm(utils_dtype, n0 * n1, ref_dl_mat, dl_mat, &ref_dl_mat_fnorm, &dl_mat_relerr);
    k_mat_relerr  /= ref_k_mat_fnorm;
    dl_mat_relerr /= ref_dl_mat_fnorm;
    printf("Gaussian kernel  relerr: %.3e, %s\n", k_mat_relerr,  (k_mat_relerr  < reltol) ? "PASSED" : "FAILED");
    printf("Gaussian dkernel relerr: %.3e, %s\n", dl_mat_relerr, (dl_mat_relerr < reltol) ? "PASSED" : "FAILED");
    test_passed = test_passed && (k_mat_relerr < reltol) && (dl_mat_relerr < reltol);

    // Test Matern 3/2 kernel and its derivate
    for (int i = 0; i < n0 * n1; i++) ref_k_mat[i] = static_cast<VT>(bd_ptr[i]);
    bd_ptr += n0 * n1;
    for (int i = 0; i < n0 * n1; i++) ref_dl_mat[i] = static_cast<VT>(bd_ptr[i]);
    bd_ptr += n0 * n1;
    st = get_wtime_sec();
    matern32_krnl_grad(
        n0, n0, c0, n1, n1, c1,
        param, n0, val_type,
        1, k_mat, 1, dl_mat
    );
    et = get_wtime_sec();
    printf("matern32_krnl_grad() time: %.3f s\n", et - st);
    calc_err_2norm(utils_dtype, n0 * n1, ref_k_mat,  k_mat,  &ref_k_mat_fnorm,  &k_mat_relerr);
    calc_err_2norm(utils_dtype, n0 * n1, ref_dl_mat, dl_mat, &ref_dl_mat_fnorm, &dl_mat_relerr);
    k_mat_relerr  /= ref_k_mat_fnorm;
    dl_mat_relerr /= ref_dl_mat_fnorm;
    printf("Matern 3/2 kernel  relerr: %.3e, %s\n", k_mat_relerr,  (k_mat_relerr  < reltol) ? "PASSED" : "FAILED");
    printf("Matern 3/2 dkernel relerr: %.3e, %s\n", dl_mat_relerr, (dl_mat_relerr < reltol) ? "PASSED" : "FAILED");
    test_passed = test_passed && (k_mat_relerr < reltol) && (dl_mat_relerr < reltol);

    // Test Matern 5/2 kernel and its derivate
    for (int i = 0; i < n0 * n1; i++) ref_k_mat[i] = static_cast<VT>(bd_ptr[i]);
    bd_ptr += n0 * n1;
    for (int i = 0; i < n0 * n1; i++) ref_dl_mat[i] = static_cast<VT>(bd_ptr[i]);
    bd_ptr += n0 * n1;
    st = get_wtime_sec();
    matern52_krnl_grad(
        n0, n0, c0, n1, n1, c1,
        param, n0, val_type,
        1, k_mat, 1, dl_mat
    );
    et = get_wtime_sec();
    printf("matern52_krnl_grad() time: %.3f s\n", et - st);
    calc_err_2norm(utils_dtype, n0 * n1, ref_k_mat,  k_mat,  &ref_k_mat_fnorm,  &k_mat_relerr);
    calc_err_2norm(utils_dtype, n0 * n1, ref_dl_mat, dl_mat, &ref_dl_mat_fnorm, &dl_mat_relerr);
    k_mat_relerr  /= ref_k_mat_fnorm;
    dl_mat_relerr /= ref_dl_mat_fnorm;
    printf("Matern 5/2 kernel  relerr: %.3e, %s\n", k_mat_relerr,  (k_mat_relerr  < reltol) ? "PASSED" : "FAILED");
    printf("Matern 5/2 dkernel relerr: %.3e, %s\n", dl_mat_relerr, (dl_mat_relerr < reltol) ? "PASSED" : "FAILED");
    test_passed = test_passed && (k_mat_relerr < reltol) && (dl_mat_relerr < reltol);

    if (test_passed) printf("Test passed\n\n");
    else printf("Test failed\n\n");

    free(bd_data);
    free(c0);
    free(c1);
    free(k_mat);
    free(dl_mat);
    free(ref_k_mat);
    free(ref_dl_mat);
    return test_passed;
}

int main(const int argc, const char **argv)
{
    if (argc < 5)
    {
        printf("Usage: %s n0 n1 dim bd_file\n", argv[0]);
        printf("n0, n1  : Number of points in the 1st/2nd point set\n");
        printf("dim     : Point dimension\n");
        printf("bd_file : Binary file containing the input and reference output data, \n");
        printf("          generated using gen_kernels_test_bin.m\n");
        return 1;
    }

    int fp32_passed = test_function<float>(argc, argv);
    int fp64_passed = test_function<double>(argc, argv);
    int test_passed = fp32_passed && fp64_passed;
    printf("Are all tests passed? %s\n\n", test_passed ? "YES" : "NO");
    
    return (1 - test_passed);
}