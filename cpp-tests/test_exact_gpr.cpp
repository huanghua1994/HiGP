#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>

#include "gp/gp.h"
#include "utils.h"

using VT = double;

int main(int argc, char **agrv)
{
    if (argc < 6)
    {
        printf("Usage: %s n_train n_pred l f s\n", agrv[0]);
        printf("n_train : Number of 1D points in the training set\n");
        printf("n_pred  : Number of 1D points in the prediction set\n");
        printf("krnl : f^2 * K(X, X, l) + s * I\n");
        return 1;
    }
    int pt_dim  = 1;
    int n_train = atoi(agrv[1]);
    int n_pred  = atoi(agrv[2]);
    int n_all   = n_train + n_pred;
    VT l = atof(agrv[3]);
    VT f = atof(agrv[4]);
    VT s = atof(agrv[5]);
    VT krnl_param[4] = {(VT) pt_dim, l, f, s};

    VT *X_all = (VT *) malloc(sizeof(VT) * n_all * pt_dim);
    VT *Y_all = (VT *) malloc(sizeof(VT) * n_all);
    srand(19241112);
    for (int i = 0; i < n_all; i++)
    {
        VT noise = 0.0;
        for (int k = 0; k < 10; k++) noise += (VT) rand() / (VT) RAND_MAX;
        noise = (noise - 5.0) / 5.0 * 0.2;
        X_all[i] = (VT) rand() / (VT) RAND_MAX;
        Y_all[i] = std::sin(2.0 * M_PI * X_all[i]) * std::exp(X_all[i]) + 2.0 * X_all[i] + noise;
    }
    VT *X_train = X_all, *X_pred = X_all + n_train;
    VT *Y_train = Y_all, *Y_pred = Y_all + n_train;

    int val_type = (sizeof(VT) == sizeof(double)) ? VAL_TYPE_DOUBLE : VAL_TYPE_FLOAT;
    int krnl_id = KERNEL_ID_GAUSSIAN;
    int nnt_id = NNT_SOFTPLUS;
    VT L_grads[4] = {0, 0, 0, 0};
    printf("Kernel parameters: l = %.2f, f = %.2f, s = %.4e\n", krnl_param[1], krnl_param[2], krnl_param[3]);
    exact_gpr_loss_compute(
        val_type, nnt_id, krnl_id, (const void *) &krnl_param[0],
        n_train, pt_dim, X_train, n_all, 
        Y_train, (void *) &L_grads[0], (void *) &L_grads[1]
    );
    printf("exact_gpr_loss_compute done\n");
    dump_binary("egpr_L.bin", L_grads, sizeof(VT) * 4);

    VT *stddev = (VT *) malloc(sizeof(VT) * n_pred);
    exact_gpr_predict(
        val_type, nnt_id, krnl_id, &krnl_param[0], 
        n_train, pt_dim, X_train, n_all, 
        Y_train, n_pred, X_pred, n_all,
        Y_pred, stddev
    );

    char X_fname[64], Y_fname[64], sd_fname[64];
    sprintf(X_fname, "egp_X_%d-%d.bin", n_train, n_pred);
    sprintf(Y_fname, "egp_Y_%d-%d.bin", n_train, n_pred);
    sprintf(sd_fname, "egp_sd_%d.bin", n_pred);
    dump_binary(X_fname, X_all, sizeof(VT) * n_all * pt_dim);
    dump_binary(Y_fname, Y_all, sizeof(VT) * n_all);
    dump_binary(sd_fname, stddev, sizeof(VT) * n_pred);

    printf("Run this MATLAB function in the current folder to check results:\n");
    printf("include_all\n");
    printf(
        "check_test_exact_gpr(%d, %d, %.2f, %.2f, %.4e, \'softplus\')\n\n",
        n_train, n_pred, krnl_param[1], krnl_param[2], krnl_param[3]
    );
    
    free(X_all);
    free(Y_all);
    free(stddev);
    return 0;
}
