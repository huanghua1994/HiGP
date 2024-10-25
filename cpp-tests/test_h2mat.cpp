#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <type_traits>

#include "common.h"
#include "utils.h"
#include "kernels/kernels.h"
#include "h2mat/h2mat.h"
#include "dense_kernel_matrix.h"

template <typename VT>
int test_function(const int argc, const char **argv)
{
    const int is_float  = std::is_same<VT, float>::value;
    const int is_double = std::is_same<VT, double>::value;

    if (is_double) printf("========== Test double precision ==========\n");
    if (is_float)  printf("========== Test single precision ==========\n");

    val_type_t val_type = is_double ? VAL_TYPE_DOUBLE : VAL_TYPE_FLOAT;
    utils_dtype_t utils_dtype = is_double ? UTILS_DTYPE_FP64 : UTILS_DTYPE_FP32;

    int npt     = atoi(argv[1]);
    int pt_dim  = atoi(argv[2]);
    int krnl_id = atoi(argv[3]);
    VT  l       = atof(argv[4]);
    VT  f       = atof(argv[5]);
    VT  s       = atof(argv[6]);
    VT  reltol  = atof(argv[7]);
    VT *coord = (VT *) malloc(sizeof(VT) * npt * pt_dim);
    if (argc >= 9)
    {
        FILE *inf = fopen(argv[8], "rb");
        fread(coord, sizeof(VT), npt * pt_dim, inf);
        fclose(inf);
    } else {
        VT prefac = std::pow(static_cast<VT>(npt), 1.0 / static_cast<VT>(pt_dim));
        for (int i = 0; i < npt * pt_dim; i++)
        {
            coord[i] = (VT) rand() / RAND_MAX;
            coord[i] *= prefac;
        }
    }

    if (pt_dim > 3)
    {
        printf("H2 matrix construction only supports 1D, 2D, and 3D points\n");
        free(coord);
        return 1;
    }

    if (is_float && reltol < 1e-5)
    {
        printf("%e reltol is too small for single precision test, reset it to 1e-5\n", reltol);
        reltol = 1e-5;
    }

    if (krnl_id == 1)  printf("Using Gaussian kernel\n");
    if (krnl_id == 2)  printf("Using Matern 3/2 kernel\n");
    if (krnl_id == 3)  printf("Using Matern 5/2 kernel\n");
    if (krnl_id == 99) printf("Using custom kernel\n");
    printf("Kernel parameters: l, f, s = %.3f, %.3f, %.6e\n", l, f, s);

    int leaf_nmax = 400;
    VT leaf_emax = 0;
    octree_p octree = NULL;
    octree_build(npt, pt_dim, val_type, coord, leaf_nmax, (const void *) &leaf_emax, &octree);
    printf("Octree has %d nodes, %d levels\n", octree->n_node, octree->n_level);

    VT *dnoise = (VT *) malloc(sizeof(VT) * npt);
    for (int i = 0; i < npt; i++) dnoise[i] = 0.01 * ((VT) rand() / RAND_MAX);

    double st, et;
    VT param[4] = {(VT) pt_dim, l, f, s};
    ss_h2mat_p ss_h2mat = NULL;
    st = get_wtime_sec();
    ss_h2mat_init(octree, (void *) &param[0], dnoise, krnl_id, 1, (void *) &reltol, &ss_h2mat);
    et = get_wtime_sec();
    printf("K(X, X, l) and dK(X, X, l) / dl ss_h2mat_init       time = %.3f s\n", et - st);

    dense_krnl_mat_p dkmat = NULL;
    st = get_wtime_sec();
    dense_krnl_mat_init(
        npt, npt, coord, npt, npt, coord, 
        param, dnoise, krnl_id, val_type, &dkmat
    );
    et = get_wtime_sec();
    printf("K(X, X, l) and dK(X, X, l) / dl dense_krnl_mat_init time = %.3f s\n", et - st);

    int nvec = 8;
    VT *B      = (VT *) malloc(sizeof(VT) * npt * nvec);
    VT *h2_B   = (VT *) malloc(sizeof(VT) * npt * nvec);
    VT *h2_dlB = (VT *) malloc(sizeof(VT) * npt * nvec);
    VT *h2_dfB = (VT *) malloc(sizeof(VT) * npt * nvec);
    VT *h2_dsB = (VT *) malloc(sizeof(VT) * npt * nvec);
    VT *dk_B   = (VT *) malloc(sizeof(VT) * npt * nvec);
    VT *dk_dlB = (VT *) malloc(sizeof(VT) * npt * nvec);
    VT *dk_dfB = (VT *) malloc(sizeof(VT) * npt * nvec);
    VT *dk_dsB = (VT *) malloc(sizeof(VT) * npt * nvec);
    for (int i = 0; i < npt * nvec; i++)
        B[i] = (VT) rand() / RAND_MAX - 0.5;
    
    st = get_wtime_sec();
    ss_h2mat_grad_matmul(ss_h2mat, nvec, B, npt, h2_B, h2_dlB, h2_dfB, h2_dsB, npt);
    et = get_wtime_sec();
    printf("ss_h2mat_grad_matmul       time = %.3f s\n", et - st);

    st = get_wtime_sec();
    dense_krnl_mat_grad_matmul(dkmat, nvec, B, npt, dk_B, dk_dlB, dk_dfB, dk_dsB, npt);
    et = get_wtime_sec();
    printf("dense_krnl_mat_grad_matmul time = %.3f s\n", et - st);

    VT ref_fnorm = 0, relerr = 0, item_passed = 0, test_passed = 1;
    reltol *= 3;   // The H2 matvec error can be a little larger than the prescribed tolerance

    calc_err_2norm(utils_dtype, npt * nvec, dk_B,   h2_B,   &ref_fnorm, &relerr);
    relerr /= ref_fnorm;
    item_passed = (relerr < reltol);
    test_passed = test_passed && item_passed;
    printf("K(X, X, l)        H2 matmul F-norm relerr = %e, %s\n", relerr, item_passed ? "PASSED" : "FAILED");

    calc_err_2norm(utils_dtype, npt * nvec, dk_dlB, h2_dlB, &ref_fnorm, &relerr);
    relerr /= ref_fnorm;
    item_passed = (relerr < reltol);
    test_passed = test_passed && item_passed;
    printf("d K(X, X, l) / dl H2 matmul F-norm relerr = %e, %s\n", relerr, item_passed ? "PASSED" : "FAILED");

    calc_err_2norm(utils_dtype, npt * nvec, dk_dfB, h2_dfB, &ref_fnorm, &relerr);
    relerr /= ref_fnorm;
    item_passed = (relerr < reltol);
    test_passed = test_passed && item_passed;
    printf("d K(X, X, l) / df H2 matmul F-norm relerr = %e, %s\n", relerr, item_passed ? "PASSED" : "FAILED");

    calc_err_2norm(utils_dtype, npt * nvec, dk_dsB, h2_dsB, &ref_fnorm, &relerr);
    relerr /= ref_fnorm;
    item_passed = (relerr < reltol);
    test_passed = test_passed && item_passed;
    printf("d K(X, X, l) / ds H2 matmul F-norm relerr = %e, %s\n", relerr, item_passed ? "PASSED" : "FAILED");

    if (test_passed) printf("Test passed\n\n");
    else printf("Test failed\n\n");

    free(dnoise);
    free(B);
    free(h2_B);
    free(h2_dlB);
    free(h2_dfB);
    free(h2_dsB);
    free(dk_B);
    free(dk_dlB);
    free(dk_dfB);
    free(dk_dsB);
    dense_krnl_mat_free(&dkmat);
    ss_h2mat_free(&ss_h2mat);
    octree_free(&octree);
    free(coord);
    return test_passed;
}

int main(const int argc, const char **argv)
{
    if (argc < 8)
    {
        printf("Usage: %s n_point pt_dim krnl_id l f s reltol coord_bin (optional)\n", argv[0]);
        printf("  n_point   : Number of points\n");
        printf("  pt_dim    : Point dimension\n");
        printf("  krnl_id   : Kernel ID, 1 for Gaussian, 2 for Matern 3/2, 3 for Matern 5/2\n");
        printf("  kernel    : f^2 * K(X, X, l) + s * I\n");
        printf("  reltol    : Relative tolerance for H2 construction\n");
        printf("  coord_bin : Binary file containing point coordinates, col-major, size n_point * pt_dim\n");
        return 1;
    }

    int fp32_passed = test_function<float>(argc, argv);
    int fp64_passed = test_function<double>(argc, argv);
    int test_passed = fp32_passed && fp64_passed;
    printf("Are all tests passed? %s\n\n", test_passed ? "YES" : "NO");
    
    return (1 - test_passed);
}
