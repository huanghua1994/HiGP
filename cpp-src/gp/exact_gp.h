#ifndef __EXACT_GP__
#define __EXACT_GP__

#ifdef __cplusplus
extern "C" {
#endif

// Compute GP regression loss and its derivatives w.r.t. l, f, and s using exact solve
// Input parameters:
//   val_type   : Data type of coordinate and kernel values, 0 for double, 1 for float
//   nnt_id     : Non-negative transform ID, 0 for softplus, 1 for exp, 2 for sigmoid
//   krnl_id    : Kernel ID, 1 for Gaussian, 2 for Matern32
//   param      : Size 4, kernel parameters [dim, l, f, s], where [l, f, s] are BEFORE non-negative transformed
//   n_train    : Number of training points
//   pt_dim     : Dimension of training points
//   X_train    : Size ldX * dim, col-major, each row is a training point
//   ldX        : Leading dimension of X_train, >= n_train
//   Y_train    : n_train, correct training outputs
// Output parameters:
//   *L     : GP loss
//   L_grad : Size 3, GP loss derivatives w.r.t. f, l, and s
void exact_gpr_loss_compute(
    const int val_type, const int nnt_id, const int krnl_id, const void *param, 
    const int n_train, const int pt_dim, const void *X_train, const int ldX, 
    const void *Y_train, void *L, void *L_grad
);

// Compute GP classification loss and its derivatives w.r.t. [l, f, s] using exact solve
// Most parameters are the same as exact_gpr_loss_compute(), except for the following:
// Input parameters:
//   params   : Size n_class * 3, col-major, each row is the [l, f, s] 
//              kernel parameters (NOT non-negative transformed) for a class
//   Y_train  : Size n_train, correct training labels, should range from [0, n_class - 1]
//   n_class  : Number of classes
// Output parameter:
//   L_grads : Size n_class * 3, col-major, each row is the GP loss 
//             derivatives w.r.t. [l, f, s]
void exact_gpc_loss_compute(
    const int val_type, const int nnt_id, const int krnl_id, const void *params, 
    const int n_train, const int pt_dim, const void *X_train, const int ldX, 
    const int *Y_train, const int n_class, void *L, void *L_grads
);

// GP regression prediction with a given kernel and its parameters using exact solve
// Input parameters:
//   (First 9 parameters are the same as exact_gp_loss_compute)
//   n_pred : Number of prediction points
//   X_pred : Size ldXp * pt_dim, col-major, each row is a prediction point
//   ldXp   : Leading dimension of X_pred, >= n_pred
// Output parameters:
//   Y_pred : Size n_pred, mean of predicted outputs at X_pred
//   stddev : Size n_pred, standard deviation of predicted outputs at X_pred
void exact_gpr_predict(
    const int val_type, const int nnt_id, const int krnl_id, const void *param, 
    const int n_train, const int pt_dim, const void *X_train, const int ldX, 
    const void *Y_train, const int n_pred, const void *X_pred, const int ldXp, 
    void *Y_pred, void *stddev
);

// GP classification prediction with a given kernel and its parameters using exact solve
// Most parameters are the same as exact_gpr_predict(), except for the following:
// Input parameters:
//   n_class  : Number of classes
//   n_sample : Number of random sampling vectors for probability calculation
//   params   : Size n_class * 3, col-major, each row is the [l, f, s]
//   Y_train  : Size n_train, correct training labels, should range from [0, n_class - 1]
// Output parameters:
//   Y_pred   : Size n_pred, predicted labels at X_pred
//   Y_pred_c : Size n_pred * n_class, predicted value of each class at X_pred
//   probab   : Size n_pred * n_class, col-major, predicted probabilities of each class at X_pred
void exact_gpc_predict(
    const int val_type, const int nnt_id, const int krnl_id, const int n_class,
    const int n_sample, const void *params, const int n_train, const int pt_dim, 
    const void *X_train, const int ldX, const int *Y_train, const int n_pred, 
    const void *X_pred, const int ldXp, int *Y_pred, void *Y_pred_c, void *probab
);

#ifdef __cplusplus
}
#endif

#endif
