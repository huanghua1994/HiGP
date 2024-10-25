#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <type_traits>

#include "dense_kernel_matrix.h"
#include "solvers/solvers.h"
#include "h2mat/h2mat.h"
#include "utils.h"

template <typename VT>
int test_function(const int argc, const char **argv)
{
    const int is_float  = std::is_same<VT, float>::value;
    const int is_double = std::is_same<VT, double>::value;

    if (is_double) printf("========== Test double precision ==========\n");
    if (is_float)  printf("========== Test single precision ==========\n");

    val_type_t val_type = is_double ? VAL_TYPE_DOUBLE : VAL_TYPE_FLOAT;
    utils_dtype_t utils_dtype = is_double ? UTILS_DTYPE_FP64 : UTILS_DTYPE_FP32;

    int m    = atoi(argv[1]);
    int dim  = atoi(argv[2]);
    int nvec = atoi(argv[3]);
    VT  l    = atof(argv[4]);
    VT  f    = atof(argv[5]);
    VT  s    = atof(argv[6]);

    srand(19241112);
    VT *coord = (VT *) malloc(sizeof(VT) * m * dim);
    VT scale = pow((VT) m, 1.0 / (VT) dim);
    for (int i = 0; i < m * dim; i++) 
        coord[i] = ((VT) rand() / (VT) RAND_MAX) * scale;
    printf("%d points in %dD cube of size %.3f\n", m, dim, scale);

    int krnl_id = KERNEL_ID_GAUSSIAN;
    VT param[4] = {(VT) dim, l, f, s};
    printf("Use kernel: %s\n", krnl_id == 1 ? "Gaussian" : "matern32");
    printf("param: l = %.2f, f = %.2f, s = %.4e\n", l, f, s);

    int leaf_nmax = 400;
    VT leaf_emax = 0;
    octree_p octree = NULL;
    ss_h2mat_p ss_h2mat = NULL;
    dense_krnl_mat_p dkmat = NULL;
    matmul_fptr A_mm = NULL;
    void *A_ptr = NULL;
    double st, et;
    if (dim <= 3)
    {
        octree_build(m, dim, val_type, coord, leaf_nmax, (const void *) &leaf_emax, &octree);
        VT h2_reltol = is_double ? 1e-10 : 1e-5;
        st = get_wtime_sec();
        ss_h2mat_init(octree, (void *) &param[0], NULL, krnl_id, 1, (void *) &h2_reltol, &ss_h2mat);
        et = get_wtime_sec();
        printf("H2 matrix build time = %.2f s\n", et - st);
        A_mm = (matmul_fptr) ss_h2mat_krnl_matmul;
        A_ptr = (void *) ss_h2mat;
    } else {
        st = get_wtime_sec();
        dense_krnl_mat_init(
            m, m, coord, m, m, coord, 
            param, NULL, krnl_id, val_type, &dkmat
        );
        dense_krnl_mat_populate(dkmat);
        et = get_wtime_sec();
        printf("Dense kernel matrix build time = %.2f s\n", et - st);
        A_mm = (matmul_fptr) dense_krnl_mat_krnl_matmul;
        A_ptr = (void *) dkmat;
    }

    int glr_rank = 100, fsai_npt = 50;
    printf("glr_rank = %d, fsai_npt = %d\n", glr_rank, fsai_npt);
    afn_precond_p ap = NULL;
    st = get_wtime_sec();
    int need_grad = 0; 
    int npt_s = (500 < m) ? 500 : m;
    afn_precond_build(
        val_type, krnl_id, (void *) &param[0], NULL, 
        m, dim, (const void *) coord, m, 
        npt_s, glr_rank, fsai_npt,
        octree, need_grad, &ap
    );
    et = get_wtime_sec();
    printf("afn_precond_build() time = %.2f s\n", et - st);
    printf("AFN estimated rank = %d, K11 size = %d\n", ap->est_rank, ap->n1);
    printf("AFN falls back to Nys : %d\n", ap->is_nys);
    matmul_fptr invM_mm = (matmul_fptr) afn_precond_apply;
    void *invM_ptr = (void *) ap;

    VT *B = (VT *) malloc(sizeof(VT) * m * nvec);
    VT *X = (VT *) malloc(sizeof(VT) * m * nvec);
    VT *R = (VT *) malloc(sizeof(VT) * m * nvec);
    for (int i = 0; i < m * nvec; i++)
    {
        B[i] = (VT) rand() / (VT) RAND_MAX - 0.5;
        X[i] = 0.0;
    }

    VT reltol = 1e-6, reltol0 = 1e-3;
    int max_iter = 1000, iter0 = 0;
    int *iters = (int *) malloc(sizeof(int) * nvec);
    // Make X(:, 1) converge faster in bpcg
    st = get_wtime_sec();
    bpcg(
        m, 1, val_type, max_iter, &reltol0, 
        A_mm, A_ptr, invM_mm, invM_ptr,
        B, m, X, m, &iter0
    );
    et = get_wtime_sec();
    printf("X(:, 0) used %d iters to converge to %.3e, used %.2f s\n", iter0, reltol0, et - st);
    st = get_wtime_sec();
    bpcg(
        m, nvec, val_type, max_iter, &reltol, 
        A_mm, A_ptr, invM_mm, invM_ptr,
        B, m, X, m, iters
    );
    et = get_wtime_sec();
    printf("BPCG for %d vectors used %.2f s\n", nvec, et - st);
    A_mm(A_ptr, nvec, X, m, R, m);
    for (int j = 0; j < nvec; j++)
    {
        VT *Rj = R + j * m;
        VT *Bj = B + j * m;
        VT Bj_2norm = 0.0, err_2norm = 0.0;
        calc_err_2norm(utils_dtype, m, Bj, Rj, &Bj_2norm, &err_2norm);
        printf("RHS %d: iter = %3d, relerr = %.3e\n", j, iters[j], err_2norm);
    }
    printf("X(:, 0) total iters = %d + %d = %d\n", iter0, iters[0], iter0 + iters[0]);

    int min_iter = iters[0];
    for (int i = 1; i < nvec; i++) min_iter = (iters[i] < min_iter) ? iters[i] : min_iter;
    int test_passed = (iters[0] == min_iter) ? 1 : 0;
    if (test_passed) printf("Test passed\n\n");
    else printf("Test failed\n\n");

    free(coord);
    free(B);
    free(X);
    free(R);
    free(iters);
    return test_passed;
}

int main(const int argc, const char **argv)
{
    if (argc < 7)
    {
        printf("Usage: %s m dim nvec l f s\n", argv[0]);
        printf("m    : Number of points\n");
        printf("dim  : Point dimension\n");
        printf("nvec : Number of RHS vectors to solve\n");
        printf("krnl : f^2 * K(X, X, l) + s * I\n");
        return 1;
    }

    int fp32_passed = test_function<float>(argc, argv);
    int fp64_passed = test_function<double>(argc, argv);
    int test_passed = fp32_passed && fp64_passed;
    printf("Are all tests passed? %s\n\n", test_passed ? "YES" : "NO");
    
    return (1 - test_passed);
}

