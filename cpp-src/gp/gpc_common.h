#ifndef __GPC_COMMON_H__
#define __GPC_COMMON_H__

#ifdef __cplusplus
extern "C" {
#endif

// Convert training labels to dnoises and RHS vectors
// Input parameter:
//   val_type : Data type of Y_train, 0 for double, 1 for float
//   n_train  : Number of training points
//   n_class  : Number of classes
//   Y_train  : Size n_train, range [0, n_class-1], label of each training point
// Output parameters:
//   *dnoises : Size n_train * n_class, col-major, each column is a dnoise vector
//   *Ys      : Size n_train * n_class, col-major, each column is a RHS vector
void gpc_process_label(
    const int val_type, const int n_train, const int n_class, 
    const int *Y_train, void **dnoises, void **Ys
);

// Draw nvec random dim-dimensional vectors from a multivariate Gaussian distribution
// Input parameters:
//   val_type : Data type of mu, S, and X, 0 for double, 1 for float
//   dim      : Dimension of the Gaussian distribution
//   mu       : Size dim, mean vector of the Gaussian distribution
//   S        : Size dim * dim, col-major, symmetric covariance matrix of the Gaussian distribution
//   nvec     : Number of vectors to draw
// Output parameter:
//   X : Size dim * nvec, col-major, each column is a random vector
void mvnrnd(const int val_type, const int dim, const void *mu, const void *S, const int nvec, void *X);

// Compute predicted labels and probabilities for GP classification
// Input parameters:
//   val_type : Data type of samples, Y_pred_c, and probab, 0 for double, 1 for float
//   n_class  : Number of classes
//   n_pred   : Number of prediction points
//   n_sample : Number of random sampling vectors for probability calculation
//   samples  : Size n_pred * n_sample * n_class, col-major
//   Y_pred_c : Size n_pred * n_class, predicted value of each class at X_pred
// Output parameters:
//   samples : Will be overwritten by exp(samples)
//   Y_pred  : Size n_pred, predicted labels at X_pred
//   probab  : Size n_pred * n_class, col-major, predicted probabilities of each class at X_pred
void gpc_pred_probab(
    const int val_type, const int n_class, const int n_pred, const int n_sample, 
    void *samples, const void *Y_pred_c, int *Y_pred, void *probab
);

#ifdef __cplusplus
}
#endif

#endif
