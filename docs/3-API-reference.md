# 3. API Reference

Note: we suggest using default values for optional inputs for general usage.

## 3.1 Definition of hyperparameters

We use three hyperparameters $l$, $f$, and $s$ in HiGP.

$l$ is the length scale parameter in kernel functions, for example the Gaussian kernel $\kappa(x,y) = \exp(-\|x-y\|_2^2 / (2 l^2) )$.

$f$ and $s$ are the scaling factors. For training data ${\mathbf{X}} \in {\mathbb{R}}^{n \times d}$, noisy training observations ${\mathbf{y}} \in {\mathbb{R}}^{n}$, and testing data $`{\mathbf{X}}_{*} \in {\mathbb{R}}^{m \times d}`$, a standard GP model assumes that the noise-free test target values $`y_* \in R^{m}`$ follow the joint distribution:

```math
\begin{bmatrix}
    {\mathbf{y}} \\
    {\mathbf{y}}_{*} 
\end{bmatrix}
\sim
{\mathcal{N}} 
\begin{pmatrix}
    \mathbf{0},
    f^2
    \begin{bmatrix}
        \kappa({\mathbf{X}},{\mathbf{X}}) + s \mathbf{I} & \kappa({\mathbf{X}},{\mathbf{X}}_{*} )\\ \kappa({\mathbf{X}}_{*} ,{\mathbf{X}}) & \kappa({\mathbf{X}}_{*} ,{\mathbf{X}}_{*} )
    \end{bmatrix}
\end{pmatrix}.
```

By definition, $l$, $f$, and $s$ should all be positive. HiGP uses a technique to handle this: the optimizer updates $l$, $f$, and $s$ without constraint (these values can be zero or negative), and $l$, $f$, and $s$ will be transformed using the [softplus function](https://pytorch.org/docs/stable/generated/torch.nn.Softplus.html) $\theta_{real} = \text{softplus}(\theta), \ \theta \in \{l, f, s\}$ internally in HiGP. All hyperparameters in the APIs in Section 3 are $\theta$ instead of $\theta_{real}$.

## 3.2 Class `higp.GPRModel`

GP regression model for PyTorch.

`__init__()`: Initialize the model

* Input:
  * `gprproblem` : GPRProblem object
* Optional inputs (default values):
  * `l, f, s (0, 0, 0)` : Hyperparameters (before transformation)
  * `dtype (torch.float32)` : torch datatype

`calc_loss_grad()`: Calculate the loss and grad of the parameters and set the gradients of hyperparameters

* Optional input (default values):
  * `scale (-1)` : Scale for loss; if < 0, will use $1 / N$ (N is the number of training data points). We suggest using the default scaling factor.

`get_params()`: Return model parameters

* Output:
  * `np` : NumPy 1D array of length 3, $[l, f, s]$

## 3.3 Class `higp.GPCModel`

GP classification model for PyTorch.

`__init__()`: Initialize the model

* Input:
  * `gpcproblem` : GPCProblem object
  * `num_classes` : Number of classes
* Optional inputs (default values):
  * `l, f, s (0, 0, 0)` : Hyperparameters (before transformation)
  * `dtype (torch.float32)` : torch datatype

`calc_loss_grad()`: Calculate the loss and grad of the parameters and set the gradients of hyperparameters

* Optional input (default values):
  * `scale (-1)` : Scale for loss; if < 0, will use $1 / N$ (N is the number of training data points). We suggest using the default scaling factor.

`get_params()`: Return model parameters

* Output:
  * `np` : NumPy 1D array of length `3 * num_classes`, $[l_1, f_1, s_1, l_2, f_2, s_2, ...]$

## 3.4 Method `gpr_torch_minimize()`

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

## 3.5 Method `gpc_torch_minimize()`

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

## 3.6 Method `ezgpr_torch()`

Easy to use GP regression interface with PyTorch using Adam optimizer.

* Inputs:
  * `train_x` : PyTorch tensor / row-major NumPy array, training data of size `d`-by-`N1` (or array of size `N1` if `d = 1`)
  * `train_y` : PyTorch tensor / row-major NumPy array, training labels of size `N1`
  * `test_x`  : PyTorch tensor / row-major NumPy array, testing data of size `d`-by-`N2` (or array of size `N2` if `d = 1`)
* Optional Inputs (default value):
  * `test_y (None)`                     : PyTorch tensor / row-major NumPy array, testing labels of size `N2`, only used for RMSE calculation
  * `l_init (0.0)`                      : Initial value of `l` (before transformation)
  * `f_init (0.0)`                      : Initial value of `f` (before transformation)
  * `s_init (0.0)`                      : Initial value of `s` (before transformation)
  * `n_threads (-1)`                    : Number of threads. If negative will use the system's default
  * `exact_gp (0)`                      : Whether to use exact matrix solve in GP computation
  * `kernel_type (higp.GaussianKernel)` : Kernel type, can be `higp.GaussianKernel`, `higp.Matern32Kernel`, `higp.Matern52Kernel`, or `higp.CustomKernel`.
  * `mvtype (higp.MatvecAuto)`          : Matvec type: can be `higp.MatvecAuto`, `higp.MatvecAOT`, or `higp.MatvecOTF`
  * `afn_rank_lq (50)`                  : The rank of the AFN preconditioner for Lanczos quadrature
  * `afn_lfil_lq (0)`                   : The fill-level of the Schur complement of the AFN preconditioner for Lanczos quadrature
  * `afn_rank_pred (50)`                : The rank of the AFN preconditioner for prediction
  * `afn_lfil_pred (0)`                 : The fill-level of the Schur complement of the AFN preconditioner for prediction
  * `niter_lq (10)`                     : Number of iterations for the Lanczos quadrature
  * `nvec_lq (10)`                      : Number of vectors for the Lanczos quadrature
  * `niter_pred (500)`                  : Number of the PCG solver iterations for the prediction
  * `tol_pred (1e-5)`                   : Prediction PCG solver tolerance
  * `seed (42)`                         : Random number generator seed
  * `adam_lr (0.1)`                     : Adam optimizer learning rate
  * `adam_maxits (100)`                 : Max number of iterations for the Adam optimizer
  * `dtype_torch (torch.float32)`       : PyTorch datatype
* Outputs:
  * `pred` : structure, containing two members
    * `pred.prediction_mean`   : NumPy array, size `N2`, prediction mean values
    * `pred.prediction_stddev` : NumPy array, size `N2`, prediction standard deviation
