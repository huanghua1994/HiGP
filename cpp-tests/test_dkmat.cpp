#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <type_traits>

#include "dense_kernel_matrix.h"
#include "utils.h"

template <typename VT>
int test_function(const int argc, const char **argv)
{
    const int is_float  = std::is_same<VT, float>::value;
    const int is_double = std::is_same<VT, double>::value;

    if (is_double) printf("========== Test double precision ==========\n");
    if (is_float)  printf("========== Test single precision ==========\n");

    VT reltol = is_double ? 1e-12 : 1e-5;
    val_type_t val_type = is_double ? VAL_TYPE_DOUBLE : VAL_TYPE_FLOAT;
    utils_dtype_t utils_dtype = is_double ? UTILS_DTYPE_FP64 : UTILS_DTYPE_FP32;

    int n0   = atoi(argv[1]);
    int n1   = atoi(argv[2]);
    int dim  = atoi(argv[3]);
    int nvec = atoi(argv[4]);

    int test_passed = 1;
    double st, et;

    size_t bd_size = 3;         // l, f, s
    bd_size += n0 * dim;        // c0
    bd_size += n1 * dim;        // c1
    bd_size += n0;              // dnoise
    bd_size += n1 * nvec;       // B
    bd_size += n0 * nvec;       // K_B
    bd_size += n0 * nvec;       // dKdl_B
    bd_size += n0 * nvec;       // dKdf_B
    bd_size += n0 * nvec;       // dKds_B
    bd_size *= sizeof(double);  // Input binary are all double precision values
    double *bd_data = (double *) malloc(bd_size);
    read_binary_file(argv[5], bd_data, bd_size);

    double *bd_ptr = bd_data;
    VT l = static_cast<VT>(bd_ptr[0]);
    VT f = static_cast<VT>(bd_ptr[1]);
    VT s = static_cast<VT>(bd_ptr[2]);
    VT param[4] = {(VT) dim, l, f, s};
    bd_ptr += 3;

    VT *c0     = (VT *) malloc(sizeof(VT) * n0 * dim);
    VT *c1     = (VT *) malloc(sizeof(VT) * n1 * dim);
    VT *dnoise = (VT *) malloc(sizeof(VT) * n0);
    for (int i = 0; i < n0 * dim; i++) c0[i] = static_cast<VT>(bd_ptr[i]);
    bd_ptr += n0 * dim;
    for (int i = 0; i < n1 * dim; i++) c1[i] = static_cast<VT>(bd_ptr[i]);
    bd_ptr += n1 * dim;
    for (int i = 0; i < n0; i++) dnoise[i] = static_cast<VT>(bd_ptr[i]);
    bd_ptr += n0;

    int krnl_id = KERNEL_ID_GAUSSIAN;
    printf("Use kernel: %s\n", krnl_id == 1 ? "Gaussian" : "Matern32");
    printf("param: l = %.4f, f = %.4f, s = %.6e\n", l, f, s);

    dense_krnl_mat_p dkmat;
    dense_krnl_mat_init(
        n0, n0, c0, n1, n1, c1, 
        param, dnoise, krnl_id, val_type, &dkmat
    );
    st = get_wtime_sec();
    dense_krnl_mat_populate(dkmat);
    et = get_wtime_sec();
    printf("dense_krnl_mat_populate() done, time = %.3f s\n", et - st);

    VT *B          = (VT *) malloc(sizeof(VT) * n1 * nvec);
    VT *K_B        = (VT *) malloc(sizeof(VT) * n0 * nvec);
    VT *dKdl_B     = (VT *) malloc(sizeof(VT) * n0 * nvec);
    VT *dKdf_B     = (VT *) malloc(sizeof(VT) * n0 * nvec);
    VT *dKds_B     = (VT *) malloc(sizeof(VT) * n0 * nvec);
    VT *ref_K_B    = (VT *) malloc(sizeof(VT) * n0 * nvec);
    VT *ref_dKdl_B = (VT *) malloc(sizeof(VT) * n0 * nvec);
    VT *ref_dKdf_B = (VT *) malloc(sizeof(VT) * n0 * nvec);
    VT *ref_dKds_B = (VT *) malloc(sizeof(VT) * n0 * nvec);

    for (int i = 0; i < n1 * nvec; i++) B[i] = static_cast<VT>(bd_ptr[i]);
    bd_ptr += n1 * nvec;
    for (int i = 0; i < n0 * nvec; i++) ref_K_B[i]    = static_cast<VT>(bd_ptr[i]);
    bd_ptr += n0 * nvec;
    for (int i = 0; i < n0 * nvec; i++) ref_dKdl_B[i] = static_cast<VT>(bd_ptr[i]);
    bd_ptr += n0 * nvec;
    for (int i = 0; i < n0 * nvec; i++) ref_dKdf_B[i] = static_cast<VT>(bd_ptr[i]);
    bd_ptr += n0 * nvec;
    for (int i = 0; i < n0 * nvec; i++) ref_dKds_B[i] = static_cast<VT>(bd_ptr[i]);

    st = get_wtime_sec();
    dense_krnl_mat_grad_matmul(dkmat, nvec, B, n1, K_B, dKdl_B, dKdf_B, dKds_B, n0);
    et = get_wtime_sec();
    printf("dense_krnl_mat_grad_matmul() done, time = %.3f s\n", et - st);

    VT ref_K_B_fnorm = 0, ref_dKdl_B_fnorm = 0, ref_dKdf_B_fnorm = 0, ref_dKds_B_fnorm = 0;
    VT K_B_relerr = 0, dKdl_B_relerr = 0, dKdf_B_relerr = 0, dKds_B_relerr = 0;
    calc_err_2norm(utils_dtype, n0 * nvec, ref_K_B,    K_B,    &ref_K_B_fnorm,    &K_B_relerr);
    calc_err_2norm(utils_dtype, n0 * nvec, ref_dKdl_B, dKdl_B, &ref_dKdl_B_fnorm, &dKdl_B_relerr);
    calc_err_2norm(utils_dtype, n0 * nvec, ref_dKdf_B, dKdf_B, &ref_dKdf_B_fnorm, &dKdf_B_relerr);
    calc_err_2norm(utils_dtype, n0 * nvec, ref_dKds_B, dKds_B, &ref_dKds_B_fnorm, &dKds_B_relerr);
    K_B_relerr    /= ref_K_B_fnorm;
    dKdl_B_relerr /= ref_dKdl_B_fnorm;
    dKdf_B_relerr /= ref_dKdf_B_fnorm;
    if (ref_dKds_B_fnorm > 0) dKds_B_relerr /= ref_dKds_B_fnorm;
    printf("K_B    relerr = %.2e, %s\n", K_B_relerr,    (K_B_relerr    < reltol) ? "PASSED" : "FAILED");
    printf("dKdl_B relerr = %.2e, %s\n", dKdl_B_relerr, (dKdl_B_relerr < reltol) ? "PASSED" : "FAILED");
    printf("dKdf_B relerr = %.2e, %s\n", dKdf_B_relerr, (dKdf_B_relerr < reltol) ? "PASSED" : "FAILED");
    printf("dKds_B relerr = %.2e, %s\n", dKds_B_relerr, (dKds_B_relerr < reltol) ? "PASSED" : "FAILED");
    test_passed = test_passed && (K_B_relerr    < reltol);
    test_passed = test_passed && (dKdl_B_relerr < reltol);
    test_passed = test_passed && (dKdf_B_relerr < reltol);
    test_passed = test_passed && (dKds_B_relerr < reltol);
    
    if (test_passed) printf("Test passed\n\n");
    else printf("Test failed\n\n");

    free(bd_data);
    free(c0);
    free(c1);
    free(dnoise);
    free(B);
    free(K_B);
    free(dKdl_B);
    free(dKdf_B);
    free(dKds_B);
    free(ref_K_B);
    free(ref_dKdl_B);
    free(ref_dKdf_B);
    free(ref_dKds_B);
    dense_krnl_mat_free(&dkmat);
    return test_passed;
}

int main(const int argc, const char **argv)
{
    if (argc < 6)
    {
        printf("Usage: %s n0 n1 dim nvec bd_file\n", argv[0]);
        printf("n0, n1  : Number of points in the 1st/2nd point set\n");
        printf("dim     : Point dimension\n");
        printf("nvec    : Number of vectors to be multiplied with the kernel matrix\n");
        printf("krnl    : f^2 * K(X, X, l) + s * I\n");
        printf("bd_file : Binary file containing the input and reference output data, \n");
        printf("          generated using gen_dkmat_test_bin.m\n");
        return 1;
    }

    int fp32_passed = test_function<float>(argc, argv);
    int fp64_passed = test_function<double>(argc, argv);
    int test_passed = fp32_passed && fp64_passed;
    printf("Are all tests passed? %s\n\n", test_passed ? "YES" : "NO");

    return (1 - test_passed);
}
