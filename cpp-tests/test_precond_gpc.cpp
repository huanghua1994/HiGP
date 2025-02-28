#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "gp/gp.h"
#include "common.h"

using VT = double;

int main(int argc, char **agrv)
{
    if (argc < 7)
    {
        printf("Usage: %s n_train n_pred n_class l f s\n", agrv[0]);
        printf("n_train : Number of 1D points in the training set\n");
        printf("n_pred  : Number of 1D points in the prediction set\n");
        printf("n_class : Number of classes\n");
        printf("krnl : f^2 * K(X, X, l) + s * I\n");
        printf("For each class, the kernel parameters are [l, f, s] * (1 + 0.1 * rand())\n");
        return 1;
    }
    int pt_dim  = 1;
    int n_train = atoi(agrv[1]);
    int n_pred  = atoi(agrv[2]);
    int n_class = atoi(agrv[3]);
    int n_all   = n_train + n_pred;
    VT l = atof(agrv[4]);
    VT f = atof(agrv[5]);
    VT s = atof(agrv[6]);
    VT *params = (VT *) malloc(sizeof(VT) * n_class * 3);
    for (int i = 0; i < n_class; i++)
    {
        params[i * 3]     = l * (1.0 + 0.1 * (VT) rand() / (VT) RAND_MAX);
        params[i * 3 + 1] = f * (1.0 + 0.1 * (VT) rand() / (VT) RAND_MAX);
        params[i * 3 + 2] = s * (1.0 + 0.1 * (VT) rand() / (VT) RAND_MAX);
    }
    char params_fname[64];
    sprintf(params_fname, "pgpc_params_%d.bin", n_class);
    dump_binary(params_fname, params, sizeof(VT) * n_class * 3);

    VT  *X_all = (VT *)  malloc(sizeof(VT)  * n_all * pt_dim);
    int *Y_all = (int *) malloc(sizeof(int) * n_all);
    srand(19241112);
    for (int i = 0; i < n_all; i++)
    {
        VT noise = 0.0;
        for (int k = 0; k < 10; k++) noise += (VT) rand() / (VT) RAND_MAX;
        noise = (noise - 5.0) / 5.0 * 0.2;
        X_all[i] = (VT) rand() / (VT) RAND_MAX;
        Y_all[i] = rand() % n_class;
    }
    VT  *X_train = X_all, *X_pred = X_all + n_train;
    int *Y_train = Y_all, *Y_pred = Y_all + n_train;

    int val_type = (sizeof(VT) == sizeof(double)) ? VAL_TYPE_DOUBLE : VAL_TYPE_FLOAT;
    int Y_val_type = VAL_TYPE_INT;
    int krnl_id = KERNEL_ID_GAUSSIAN;
    int nnt_id = NNT_SOFTPLUS;
    int glr_rank = 20, fsai_npt = 10;
    int npt_s = -glr_rank-1;  // Prevent AFN from falling back to Nystrom
    int n_iter = 5, n_vec = 5;
    symm_kmat_alg_t kmat_alg = SYMM_KMAT_ALG_DENSE;
    pgp_loss_p pgp_loss = NULL;
    pgp_loss_init(
        val_type, nnt_id, n_train, pt_dim, 
        X_train, n_all, Y_train, Y_val_type, 
        npt_s, glr_rank, fsai_npt, n_iter, 
        n_vec, kmat_alg, &pgp_loss
    );
    printf("\npgp_loss_init done\n\n");

    setenv("LANQUAD_DUMP_Z", "1", 1);  // Tell lanquad() to dump Z

    VT *L_grads = (VT *) malloc(sizeof(VT) * (n_class * 3 + 1));
    precond_gpc_loss_compute(
        pgp_loss, krnl_id, n_class, (const void *) params, 
        (void *) &L_grads[0], (void *) &L_grads[1]
    );
    pgp_loss_free(&pgp_loss);
    printf("precond_gpc_loss_compute done\n");
    char L_fname[64];
    sprintf(L_fname, "pgpc_L_%d.bin", n_class);
    dump_binary(L_fname, L_grads, sizeof(VT) * (n_class * 3 + 1));

    int n_sample = 256;
    int max_iter = 500;
    VT rel_tol = 1e-10;
    setenv("PGPC_DUMP_RNDVEC", "1", 1);  // Tell precond_gpc_predict() to dump rndvec
    VT *Y_pred_c = (VT *) malloc(sizeof(VT) * n_pred * n_class);
    VT *probab   = (VT *) malloc(sizeof(VT) * n_pred * n_class);
    precond_gpc_predict(
        val_type, nnt_id, krnl_id, n_class,
        n_sample, (const void *) params, n_train, pt_dim,
        (const void *) X_train, n_all, Y_train, n_pred,
        (const void *) X_pred, n_all, npt_s, glr_rank,
        fsai_npt, max_iter, (const void *) &rel_tol, kmat_alg,
        Y_pred, (void *) Y_pred_c, (void *) probab
    );

    char X_fname[64], Y_fname[64];
    sprintf(X_fname, "pgpc_X_%d-%d.bin", n_train, n_pred);
    sprintf(Y_fname, "pgpc_Y_%d-%d.bin", n_train, n_pred);
    dump_binary(X_fname, X_all, sizeof(VT) * n_all * pt_dim);
    dump_binary(Y_fname, Y_all, sizeof(int) * n_all);

    char Yc_fname[64], pb_fname[64];
    sprintf(Yc_fname, "pgpc_Yc_%d-%d.bin", n_pred, n_class);
    sprintf(pb_fname, "pgpc_pb_%d-%d.bin", n_pred, n_class);
    dump_binary(Yc_fname, Y_pred_c, sizeof(VT) * n_pred * n_class);
    dump_binary(pb_fname, probab,   sizeof(VT) * n_pred * n_class);

    printf("Run this MATLAB function in the current folder to check results:\n");
    printf("include_all\n");
    printf(
        "check_test_precond_gpc(%d, %d, %d, %d, \'softplus\', %d, %d, %d, %d, %d, %.2e)\n\n",
        n_train, n_pred, n_class, n_sample, glr_rank, fsai_npt, n_iter, n_vec, max_iter, rel_tol
    );

    free(params);
    free(X_all);
    free(Y_all);
    free(L_grads);
    free(Y_pred_c);
    free(probab);
    return 0;
}