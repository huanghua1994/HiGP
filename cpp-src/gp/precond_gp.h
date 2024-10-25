#ifndef __PRECOND_GP_H__
#define __PRECOND_GP_H__

#include "../common.h"
#include "../h2mat/h2mat.h"
#include "../dense_kernel_matrix.h"
#include "../solvers/solvers.h"
struct pgp_loss
{
    int  val_type;      // 0 for double, 1 for float
    int  nnt_id;        // 0 for softplus, 1 for exp, 2 for sigmoid
    int  n_train;       // Number of training points
    int  pt_dim;        // Dimension of training points
    int  npt_s;         // AFN preconditioner npt_s parameter
    int  glr_rank;      // AFN preconditioner glr_rank parameter
    int  fsai_npt;      // AFN preconditioner fsai_npt parameter
    int  n_iter;        // Number of iterations to perform in the iterative solver
    int  n_vec;         // Number of test vectors used in LANQUAD
    int  kmat_alg;      // Kernel matrix algorithm and data structure
    void *X_train;      // Size n_train * pt_dim, col-major, each row is a training point
    void *Y_train;      // Size n_train, correct training outputs
    octree_p octree;    // Octree for training points
};
typedef struct pgp_loss  pgp_loss_s;
typedef struct pgp_loss *pgp_loss_p;

#ifdef __cplusplus
extern "C" {
#endif

// Kernel-independent precomputation for preconditioned GP loss computation
// Input parameters:
//   val_type       : Data type of coordinate and kernel values, 0 for double, 1 for float
//   nnt_id         : Non-negative transform ID, 0 for softplus, 1 for exp, 2 for sigmoid
//   n_train        : Number of training points
//   pt_dim         : Dimension of training points
//   X_train        : Size ldX * dim, col-major, each row is a training point
//   ldX            : Leading dimension of X_train, >= n_train
//   Y_train        : n_train, correct training outputs. For classification problem, 
//                    Y_train values should range from [0, n_class - 1]
//   Y_val_type     : Data type of training outputs, 0 for double, 1 for float, 2 for int
//   npt_s          : AFN preconditioner number of points to sample for rank estimation
//   glr_rank       : AFN preconditioner global low-rank approximation rank
//   fsai_npt       : AFN preconditioner maximum number of nonzeros in each row of the FSAI matrix
//   n_iter         : Number of iterations to perform in the iterative solver
//   n_vec          : Number of test vectors used in LANQUAD
//   kmat_alg       : Algorithm and data structure for the square kernel matrix
// Output parameters:
//   *pgp_loss : Pointer to an initialized pgp_loss struct
void pgp_loss_init(
    const int val_type, const int nnt_id, const int n_train, const int pt_dim,
    const void *X_train, const int ldX, const void *Y_train, const int Y_val_type,
    const int npt_s, const int glr_rank, const int fsai_npt, const int n_iter, 
    const int n_vec, symm_kmat_alg_t kmat_alg, pgp_loss_p *pgp_loss
);

// Free an initialized pgp_loss struct
void pgp_loss_free(pgp_loss_p *pgp_loss);

// Compute preconditioned GP regression loss and its derivatives w.r.t. [l, f, s] 
// Input parameters:
//   pgp_loss : Pointer to an initialized pgp_loss struct
//   krnl_id  : Kernel ID, 1 for Gaussian, 2 for Matern32
//   param    : Size 4, kernel parameters [dim, l, f, s], where [l, f, s] are NOT non-negative transformed
//   dnoise   : Size pgp_loss->n_train, kernel matrix diagonal noise, should be NULL for
//              GP regression calculation and will be provided when called by precond_gpc_loss_compute()
// Output parameters:
//   *L     : GP loss
//   L_grad : Size 3, GP loss derivatives w.r.t. [l, f, s]
void precond_gpr_loss_compute(pgp_loss_p pgp_loss, const int krnl_id, const void *param, void *L, void *L_grad, void *dnoise);

// Compute preconditioned GP classification loss and its derivatives w.r.t. [l, f, s] 
// Input parameters:
//   pgp_loss : Pointer to an initialized pgp_loss struct
//   krnl_id  : Kernel ID, 1 for Gaussian, 2 for Matern32
//   n_class  : Number of classes
//   params   : Size n_class * 3, col-major, each row is the [l, f, s] 
//              kernel parameters (NOT non-negative transformed) for a class
// Output parameters:
//   *L      : GP loss
//   L_grads : Size n_class * 3, col-major, each row is the GP loss 
//             derivatives w.r.t. [l, f, s]
void precond_gpc_loss_compute(
    pgp_loss_p pgp_loss, const int krnl_id, const int n_class,
    const void *params, void *L, void *L_grads
);

// Preconditioned GP regression prediction with a given kernel and its parameters
// Input parameters:
//   (First 9 parameters are the same as pgp_loss_init and precond_gpr_loss_compute)
//   n_pred   : Number of prediction points
//   X_pred   : Size ldXp * dim, col-major, each row is a prediction point
//   ldXp     : Leading dimension of X_pred, >= n_pred
//   (npt_s, glr_rank, fsai_npt are the same as pgp_loss_init)
//   max_iter : Maximum number of PCG iterations
//   rel_tol  : Pointer to the relative tolerance for PCG
//   K11_alg  : Algorithm and data structure for the n_train * n_train K11 matrix
// Output parameters:
//   Y_pred : Size n_pred, mean of predicted outputs at X_pred
//   stddev : Size n_pred, standard deviation of predicted outputs at X_pred
void precond_gpr_predict(
    const int val_type, const int nnt_id, const int krnl_id, const void *param, 
    const int n_train, const int pt_dim, const void *X_train, const int ldXt, 
    const void *Y_train, const int n_pred, const void *X_pred, const int ldXp, 
    const int npt_s, const int glr_rank, const int fsai_npt, const int max_iter, 
    const void *rel_tol, symm_kmat_alg_t K11_alg, void *Y_pred, void *stddev
);

// Preconditioned GP classification prediction with a given kernel and its parameters
// Most parameters are the same as precond_gpr_predict(), except for the following:
// Input parameters:
//   n_class  : Number of classes
//   n_sample : Number of random sampling vectors for probability calculation
//   params   : Size n_class * 3, col-major, each row is the [l, f, s]
//   Y_train  : Size n_train, correct training labels, should range from [0, n_class - 1]
// Output parameters:
//   Y_pred   : Size n_pred, predicted labels at X_pred
//   Y_pred_c : Size n_pred * n_class, predicted value of each class at X_pred
//   probab   : Size n_pred * n_class, col-major, predicted probabilities of each class at X_pred
void precond_gpc_predict(
    const int val_type, const int nnt_id, const int krnl_id, const int n_class, 
    const int n_sample, const void *params, const int n_train, const int pt_dim, 
    const void *X_train, const int ldXt, const int *Y_train, const int n_pred, 
    const void *X_pred, const int ldXp, const int npt_s, const int glr_rank, 
    const int fsai_npt, const int max_iter, const void *rel_tol, symm_kmat_alg_t K11_alg, 
    int *Y_pred, void *Y_pred_c, void *probab
);

#ifdef __cplusplus
}
#endif

#endif
