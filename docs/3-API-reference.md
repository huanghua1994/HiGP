# 3. API Reference

Note: we suggest using default values for optional inputs for general usage.

## 3.1 Class `higp.GPRModel`

GP regression model for PyTorch.

`__init__()`: Initialize the model

* Input:
  * `gprproblem` : GPRProblem object
* Optional inputs (default values):
  * `l, f, s (0, 0, 0)` : Hyperparameters (before transformation)
  * `dtype (torch.float32)` : torch datatype

`calc_loss_grad()`: Calculate the loss and grad of the parameters and set the gradients of hyperparameters

* Optional input (default values):
  * `scale (-1)` : Scale for loss; if < 0, will use $1 / N$ (N is the number of training data points) We suggest using the default scaling factor.

`get_params()`: Return model parameters

* Output:
  * `np` : NumPy 1D array of length 3, $[l, f, s]$

## 3.2 Class `higp.GPCModel`

GP classification model for PyTorch.

`__init__()`: Initialize the model

* Input:
  * `gprproblem` : GPRProblem object
  * `num_classes` : Number of classes
* Optional inputs (default values):
  * `l, f, s (0, 0, 0)` : Hyperparameters (before transformation)
  * `dtype (torch.float32)` : torch datatype

`calc_loss_grad()`: Calculate the loss and grad of the parameters and set the gradients of hyperparameters

* Optional input (default values):
  * `scale (-1)` : Scale for loss; if < 0, will use $1 / N$ (N is the number of training data points) We suggest using the default scaling factor.

`get_params()`: Return model parameters

* Output:
  * `np` : NumPy 1D array of length `3 * num_classes`, $[l_1, f_1, s_1, l_2, f_2, s_2, ...]$

## 3.3 Method `gpr_torch_minimize()`

GP regression minimization using PyTorch optimizer.

* Inputs:
  * `model` : `GPRModel` object
  * `optimizer` : PyTorch optimizer
* Optional inputs (default value):
  * `maxits (100)` : Max number of iterations
  * `scale (-1)`: Scale for loss; if < 0, will use 1 / N (N is the number of training data points)
  * `print_info (False)` : Print iteration and hyperparameters or not
* Outputs:
  * `loss_hist` : NumPy 1D array of length `maxits+1`, module loss function value after each iteration
  * `param_hist`: NumPy 2D matrix of size `maxits+1`-by-3, each row are the hyperparameters after each iteration

## 3.4 Method `gpc_torch_minimize()`

GP regression minimization using PyTorch optimizer.

* Inputs:
  * `model` : `GPRModel` object
  * `optimizer` : PyTorch optimizer
* Optional inputs (default value):
  * `maxits (100)` : Max number of iterations
  * `scale (-1)`: Scale for loss; if < 0, will use 1 / N (N is the number of training data points)
  * `print_info (False)` : Print iteration and hyperparameters or not
* Outputs:
  * `loss_hist` : NumPy 1D array of length `maxits+1`, module loss function value after each iteration
  * `param_hist`: NumPy 2D matrix of size `maxits+1`-by-`3*num_classes`, each row are the hyperparameters after each iteration

## 3.5 Method `ezgpr_torch()`

Easy to use GP regression interface with PyTorch using Adam optimizer.

* Inputs:
  * `train_x` : PyTorch tensor / row-major NumPy array, training data of size `d`-by-`N` (or array of size `N` if `d = 1`)
  * `train_y` : PyTorch tensor / row-major NumPy array, training labels of size `N`
  * `test_x`  : PyTorch tensor / row-major NumPy array, testing data of size `d`-by-`N` (or array of size `N` if `d = 1`)
  * `test_y`  : PyTorch tensor / row-major NumPy array, testing labels of size `N`
* Optional Inputs (default value):
  * `l_init (0.0)`                      : Initial value of `l` (before transformation)
  * `f_init (0.0)`                      : Initial value of `f` (before transformation)
  * `s_init (0.0)`                      : Initial value of `s` (before transformation)
  * `n_threads (-1)`                    : Number of threads. If negative will use the system's default
  * `exact_gp (0)`                      : Whether to use exact matrix solve in GP computation
  * `kernel_type (higp.GaussianKernel)` : Kernel type, can be `higp.GaussianKernel`, `higp.Matern32Kernel`, `higp.Matern52Kernel`, or `higp.CustomKernel`.
  * `mvtype (higp.MatvecAuto)`          : Matvec type: can be `higp.MatvecAuto`, `higp.MatvecAOT`, or `higp.MatvecOTF`.
  * `afn_rank_lq (50)`                  : The rank of the AFN preconditioner for Lanczos quadrature
  * `afn_lfil_lq (0)`                   : The fill-level of the Schur complement of the AFN preconditioner for Lanczos quadrature
  * `afn_rank_pred (50)`                : The rank of the AFN preconditioner forprediction
  * `afn_lfil_pred (0)`                 : The fill-level of the Schur complement of the AFN preconditioner for prediction
  * `niter_lq (10)`                     : Number of iterations for the Lanczos quadrature
  * `nvec_lq (10)`                      : Number of vectors for the Lanczos quadrature
  * `niter_pred (500)`                  : Number of the PCG solver iterations for the prediction
  * `tol_pred (1e-5)`                   : Prediction PCG solver tolerance
  * `seed (42)`                         : Random seed. If negative will not set seed.
  * `adam_lr (0.1)`                     : Adam optimizer learning rate
  * `adam_maxits (100)`                 : Max number of iterations for the Adam optimizer
  * `dtype_torch (torch.float32)`       : PyTorch datatype
* Outputs:
  * `pred_y`: NumPy array, prediction labels
  * `std_y`: Predict standard deviation
