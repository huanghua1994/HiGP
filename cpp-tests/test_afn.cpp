#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <type_traits>

#include "common.h"
#include "kernels/kernels.h"
#include "solvers/solvers.h"
#include "utils.h"

template <typename VT>
int scalar_allclose(const VT x, const VT ref_x, const VT reltol, const char *scalar_name)
{
    VT relerr = std::abs((x - ref_x) / ref_x);
    int ret = (relerr < reltol);
    if (scalar_name != NULL) printf("%s: ", scalar_name);
    printf("x = % .6e, ref_x = % .6e, relerr = % .6e, %s\n", x, ref_x, relerr, ret ? "PASSED" : "FAILED");
    return ret;
}

template <typename VT>
int test_function(const int argc, const char **argv)
{
    const int is_float  = std::is_same<VT, float>::value;
    const int is_double = std::is_same<VT, double>::value;

    if (is_double) printf("========== Test double precision ==========\n");
    if (is_float)  printf("========== Test single precision ==========\n");

    // Single precision apply() and dapply() results might have larger errors
    // since the reference results are calculated in double precision
    VT reltol = is_double ? 1e-10 : 1e-1;
    val_type_t val_type = is_double ? VAL_TYPE_DOUBLE : VAL_TYPE_FLOAT;
    utils_dtype_t utils_dtype = is_double ? UTILS_DTYPE_FP64 : UTILS_DTYPE_FP32;

    int m        = atoi(argv[1]);
    int dim      = atoi(argv[2]);
    int nvec     = atoi(argv[3]);
    int glr_rank = atoi(argv[4]);
    int fsai_npt = atoi(argv[5]);

    int test_passed = 1;
    double st, et;

    size_t bd_size = 3;         // l, f, s
    bd_size += m * dim;         // coord
    bd_size += m;               // dnoise
    bd_size += 4;               // gt and logdet
    bd_size += m * nvec;        // B
    bd_size += m * nvec;        // C
    bd_size += m * nvec * 3;    // D
    bd_size *= sizeof(double);  // Input binary are all double precision values
    double *bd_data = (double *) malloc(bd_size);
    read_binary_file(argv[6], bd_data, bd_size);

    double *bd_ptr = bd_data;
    VT l = static_cast<VT>(bd_ptr[0]);
    VT f = static_cast<VT>(bd_ptr[1]);
    VT s = static_cast<VT>(bd_ptr[2]);
    VT param[4] = {(VT) dim, l, f, s};
    bd_ptr += 3;

    VT *coord = (VT *) malloc(sizeof(VT) * m * dim);
    for (int i = 0; i < m * dim; i++) coord[i] = static_cast<VT>(bd_ptr[i]);
    bd_ptr += m * dim;

    int krnl_id = KERNEL_ID_GAUSSIAN;
    int need_grad = 1;
    int npt_s = -glr_rank-1;  // Mute rank estimation, always uses AFN instead of Nystrom
    VT *dnoise = (VT *) malloc(sizeof(VT) * m);
    for (int i = 0; i < m; i++) dnoise[i] = static_cast<VT>(bd_ptr[i]);
    bd_ptr += m;

    afn_precond_p ap;
    st = get_wtime_sec();
    afn_precond_build(
        val_type, krnl_id, (void *) &param[0], dnoise, 
        m, dim, (const void *) coord, m, 
        npt_s, glr_rank, fsai_npt,
        NULL, need_grad, &ap
    );
    et = get_wtime_sec();
    printf("afn_precond_build() done, time = %.3f s\n", et - st);

    VT *gt = (VT *) ap->gt;
    VT *logdet = (VT *) ap->logdet;
    VT ref_gt0    = static_cast<VT>(bd_ptr[0]);
    VT ref_gt1    = static_cast<VT>(bd_ptr[1]);
    VT ref_gt2    = static_cast<VT>(bd_ptr[2]);
    VT ref_logdet = static_cast<VT>(bd_ptr[3]);
    bd_ptr += 4;
    test_passed = test_passed && scalar_allclose(gt[0],     ref_gt0,    reltol, "gt[0] (l)");
    test_passed = test_passed && scalar_allclose(gt[1],     ref_gt1,    reltol, "gt[1] (f)");
    test_passed = test_passed && scalar_allclose(gt[2],     ref_gt2,    reltol, "gt[2] (s)");
    test_passed = test_passed && scalar_allclose(logdet[0], ref_logdet, reltol, "logdet   ");
    if (!test_passed) printf("Gradient trace and logdet test failed\n");

    VT *B     = (VT *) malloc(sizeof(VT) * m * nvec);
    VT *C     = (VT *) malloc(sizeof(VT) * m * nvec);
    VT *D     = (VT *) malloc(sizeof(VT) * m * nvec * 3);
    VT *ref_C = (VT *) malloc(sizeof(VT) * m * nvec);
    VT *ref_D = (VT *) malloc(sizeof(VT) * m * nvec * 3);
    for (int i = 0; i < m * nvec; i++) B[i] = static_cast<VT>(bd_ptr[i]);
    bd_ptr += m * nvec;
    for (int i = 0; i < m * nvec; i++) ref_C[i] = static_cast<VT>(bd_ptr[i]);
    bd_ptr += m * nvec;
    for (int i = 0; i < m * nvec * 3; i++) ref_D[i] = static_cast<VT>(bd_ptr[i]);

    st = get_wtime_sec();
    afn_precond_apply(ap, nvec, B, m, C, m);
    et = get_wtime_sec();
    printf("afn_precond_apply()   done, time = %.3f s\n", et - st);

    st = get_wtime_sec();
    afn_precond_dapply(ap, nvec, B, m, D, m);
    et = get_wtime_sec();
    printf("afn_precond_dapply()  done, time = %.3f s\n", et - st);

    VT ref_C_fnorm = 0, ref_D_fnorm = 0, C_relerr = 0, D_relerr = 0;
    calc_err_2norm(utils_dtype, m * nvec,     ref_C, C, &ref_C_fnorm, &C_relerr);
    calc_err_2norm(utils_dtype, m * nvec * 3, ref_D, D, &ref_D_fnorm, &D_relerr);
    C_relerr /= ref_C_fnorm;
    D_relerr /= ref_D_fnorm;
    printf("afn_precond_apply()   relative error = %.6e\n", C_relerr);
    printf("afn_precond_dapply()  relative error = %.6e\n", D_relerr);
    int passed_apply = (C_relerr < reltol) && (D_relerr < reltol);
    if (!passed_apply) printf("apply() and dapply() test failed\n");
    test_passed = test_passed && passed_apply;

    if (test_passed) printf("Test passed\n\n");
    else printf("Test failed\n\n");

    free(bd_data);
    free(coord);
    free(dnoise);
    free(B);
    free(C);
    free(D);
    free(ref_C);
    free(ref_D);
    afn_precond_free(&ap);
    return test_passed;
}

int main(const int argc, const char **argv)
{
    if (argc < 7)
    {
        printf("Usage: %s m dim nvec glr_rank fsai_npt bd_file\n", argv[0]);
        printf("m        : Number of points\n");
        printf("dim      : Point dimension\n");
        printf("nvec     : Number of vectors to apply the AFN precond\n");
        printf("krnl     : f^2 * K(X, X, l) + s * I\n");
        printf("glr_rank : Global low-rank approximation rank (K11 size)\n");
        printf("fsai_npt : Maximum number of nonzeros in each row of the FSAI matrix\n");
        printf("bd_file  : Binary file containing the input and reference output data, \n");
        printf("           generated using gen_afn_test_bin.m\n");
        return 1;
    }

    int fp32_passed = test_function<float>(argc, argv);
    int fp64_passed = test_function<double>(argc, argv);
    int test_passed = fp32_passed && fp64_passed;
    printf("Are all tests passed? %s\n\n", test_passed ? "YES" : "NO");
    
    return (1 - test_passed);
}