import numpy as np
import torch
import math
import time
import higp_cext
from scipy.optimize import minimize
from enum import Enum
from types import SimpleNamespace

class KernelType(Enum):
    GaussianKernel = 1
    Matern32Kernel = 2
    Matern52Kernel = 3
    CustomKernel = 99

GaussianKernel = KernelType.GaussianKernel.value
Matern32Kernel = KernelType.Matern32Kernel.value
Matern52Kernel = KernelType.Matern52Kernel.value
CustomKernel = KernelType.CustomKernel.value

class MatvecType(Enum):
    MatvecAuto = 0
    MatvecAOT = 1
    MatvecOTF = 2

MatvecAuto = MatvecType.MatvecAuto.value
MatvecAOT = MatvecType.MatvecAOT.value
MatvecOTF = MatvecType.MatvecOTF.value

def gpr_prediction(*args, **kwargs):
    pred = higp_cext.gpr_prediction_(*args, **kwargs)
    results = SimpleNamespace()
    results.prediction_mean = pred[0]
    results.prediction_stddev = pred[1]
    return results

def gpc_prediction(*args, **kwargs):
    pred = higp_cext.gpc_prediction_(*args, **kwargs)
    results = SimpleNamespace()
    results.prediction_mean = pred[0]
    results.prediction_label = pred[1]
    results.prediction_stddev = pred[2]
    return results

def gpr_scipy_minimize(gprproblem,
                       l = 0.0,
                       f = 0.0,
                       s = 0.0,
                       dtype = np.float32,
                       tol = 1e-05,
                       method = 'L-BFGS-B',
                       jac = True,
                       bounds = None,
                       maxiter = 20,
                       maxls = 20,
                       gtol = 1e-6,
                       disp = True):
    """
    GP regression minimization using scipy minimize
    Inputs:
        gprproblem : GPRProblem object
        l, f, s    : Hyperparameters (before transformation)
        dtype      : np datatype of gprproblem
        tol        : tolerance for scipy minimize
        method     : method for scipy minimize
        jac        : jac for scipy minimize
        bounds     : bounds for scipy minimize
        maxiter    : maxiter for scipy minimize
        maxls      : maxls for scipy minimize
        gtol       : gtol for scipy minimize
        disp       : disp for scipy minimize
    """
    def gpr_loss_func(x_init):
        loss_grad = gprproblem.grad(x_init.astype(dtype))
        loss = loss_grad[0]
        grad = loss_grad[1]
        return (loss, grad.astype(np.float64))

    result = minimize(gpr_loss_func, np.hstack([l, f, s]),
                      tol=tol, method=method, jac=jac, bounds=bounds,
                      options={'maxiter':maxiter, 'maxls':maxls, 'gtol':gtol, 'disp':disp})
    
    return result

def gpc_scipy_minimize(gpcproblem,
                       num_classes,
                       l = 0.0,
                       f = 0.0,
                       s = 0.0,
                       dtype = np.float32,
                       tol = 1e-05,
                       method = 'L-BFGS-B',
                       jac = True,
                       bounds = None,
                       maxiter = 20,
                       maxls = 20,
                       gtol = 1e-6,
                       disp = True):
    """
    GP classification minimization using scipy minimize
    Inputs:
        gpcproblem  : GPCProblem object
        num_classes : Number of classes
        l, f, s     : Hyperparameters (before transformation)
        dtype       : np datatype of gpcproblem
        tol         : tolerance for scipy minimize
        method      : method for scipy minimize
        jac         : jac for scipy minimize
        bounds      : bounds for scipy minimize
        maxiter     : maxiter for scipy minimize
        maxls       : maxls for scipy minimize
        gtol        : gtol for scipy minimize
        disp        : disp for scipy minimize
    """

    def gpc_loss_func(x_init):
        loss_grad = gpcproblem.grad(x_init.astype(dtype))
        loss = loss_grad[0]
        grad = loss_grad[1]
        return (loss, grad.astype(np.float64))
    
    if bounds is not None:
        bounds = bounds * num_classes

    result = minimize(gpc_loss_func, np.tile(np.array([l, f, s]), num_classes),
                      tol = tol, method = method, jac = jac, bounds = bounds,
                      options={'maxiter': maxiter, 'maxls': maxls, 'gtol': gtol, 'disp': disp})
    
    return result

class GPRModel(torch.nn.Module):
    """
    GP regression model for PyTorch
    __init__(): Initialize the model
        Input:
            gprproblem : GPRProblem object
        Optional inputs (default values): 
            l, f, s (0, 0, 0)       : Hyperparameters (before transformation)
            dtype (torch.float32)   : torch datatype
    calc_loss_grad(): Calculate the loss and grad of the parameters and set 
                      the gradients of hyperparameters
        Optional input (default value): 
            scale (-1) : Scale for loss; if < 0, will use 1 / N (N is the number of training data points)
    get_params(): Return model parameters
        Output:
            np : NumPy 1D array of length 3, $[l, f, s]$
    """
    def __init__(self, gprproblem, l = 0.0, f = 0.0, s = 0.0, dtype = torch.float32):
        super(GPRModel, self).__init__()
        self._gprproblem = gprproblem
        self._default_scale = torch.tensor(1.0 / gprproblem.get_n(), dtype=dtype)
        self._l = torch.nn.Parameter(torch.tensor(l, dtype = dtype))
        self._f = torch.nn.Parameter(torch.tensor(f, dtype = dtype))
        self._s = torch.nn.Parameter(torch.tensor(s, dtype = dtype))
        self._dtype = dtype
        self._dtype_np = np.float32 if dtype == torch.float32 else np.float64

    def forward(self, x):
        return x
    
    # Compute the loss and grad of the parameters and set the gradients of hyperparameters
    def calc_loss_grad(self, scale = -1.0):
        if scale < 0:
            scale = self._default_scale
        loss_grad = self._gprproblem.grad(np.hstack([self._l.item(), self._f.item(), self._s.item()]).astype(self._dtype_np))
        loss = loss_grad[0]
        grad = loss_grad[1]
        self._l.grad = torch.tensor(grad[0]) * scale
        self._f.grad = torch.tensor(grad[1]) * scale
        self._s.grad = torch.tensor(grad[2]) * scale
        return (loss - 2.0 * math.pi) * scale

    def get_params(self):
        return np.hstack([self._l.item(), self._f.item(), self._s.item()]).astype(self._dtype_np)

def gpr_torch_minimize(model, optimizer, maxits = 100, scale = -1.0, print_info = False):
    """
    GP regression minimization using PyTorch optimizer
    Inputs:
        model       : GPRModel object
        optimizer   : PyTorch optimizer
    Optional inputs (default value):
        maxits (100)        : Max number of iterations
        scale (-1)          : Scale for loss; if < 0, will use 1 / N (N is the number of training data points)
        print_info (False)  : Print iteration and hyperparameters or not
     Outputs:
        loss_hist   : NumPy 1D array of length maxits+1, module loss function value after each iteration
        param_hist  : NumPy 2D matrix of size (maxits+1)-by-3, each row are the hyperparameters after each iteration
    """
    if scale < 0:
        scale = model._default_scale
    else:
        scale = torch.tensor(scale, dtype=model._dtype, requires_grad=False)

    loss_hist = np.empty(maxits+1)
    param_hist = np.empty((maxits+1, 3))

    if print_info:
        print("Iteration (max %d), Elapsed time (sec), Loss, Hyperparameters (l, s, f, before nnt)" % maxits)

    t_start = time.time()
    for i in range(maxits):
        # Manually compute loss and set grad
        loss = model.calc_loss_grad(scale)

        # Save history
        loss_hist[i] = loss
        param_hist[i, 0] = model._l.item()
        param_hist[i, 1] = model._s.item()
        param_hist[i, 2] = model._f.item()

        # Update parameters
        optimizer.step()

        if print_info:
            t_elapsed = time.time() - t_start
            print("%d, %.2f, %.5f, %.3f, %.3f, %.3f" % (i + 1, t_elapsed, loss, model._l.item(), model._s.item(), model._f.item()))

    # save final loss and parameters
    loss = model.calc_loss_grad(scale)
    loss_hist[maxits] = loss
    param_hist[maxits, 0] = model._l.item()
    param_hist[maxits, 1] = model._s.item()
    param_hist[maxits, 2] = model._f.item()

    return loss_hist, param_hist

class GPCModel(torch.nn.Module):
    """
    GP classification model for PyTorch
    __init__(): Initialize the model
        Inputs:
            gpcproblem  : GPCProblem object, 
            num_classes : Number of classes
        Optional inputs (default values): 
            l, f, s (0, 0, 0)       : Hyperparameters (before transformation)
            dtype (torch.float32)   : torch datatype
    get_params(): Return model parameters
        Output:
            np array of parameters
    """
    def __init__(self, gpcproblem, num_classes, l = 0.0, f = 0.0, s = 0.0, dtype = torch.float32):
        super(GPCModel, self).__init__()
        self._gpcproblem = gpcproblem
        self._num_classes = num_classes
        self._params = torch.nn.Parameter(torch.tensor(np.tile(np.array([l, f, s]), num_classes), dtype = dtype))
        self._dtype = dtype
        self._default_scale = torch.tensor(1.0 / gpcproblem.get_n(), dtype=dtype)

    def forward(self, x):
        return x
    
    # This function manually set the grad of the parameters using our gp_loss_func
    def calc_loss_grad(self, scale = -1.0):
        if scale < 0:
            scale = self._default_scale
        loss_grad = self._gpcproblem.grad(self._params.detach().numpy())
        loss = loss_grad[0]
        grad = loss_grad[1]
        self._params.grad = torch.tensor(grad) * scale
        return (loss - 2.0 * math.pi) * scale
    
    def get_params(self):
        return self._params.detach().numpy()

def gpc_torch_minimize(model, optimizer, maxits = 100, scale = -1.0, print_info = False):
    """
    GP classification minimization using PyTorch optimizer
    Inputs:
        model      : GPCModel object
        optimizer  : PyTorch optimizer
    Optional inputs (default value):
        maxits (100)        : Max number of iterations
        scale (-1)          : Scale for loss. If < 0, will use 1 / N (N is the number of training data points)
        print_info (False)  : Print iteration and hyperparameters or not
    Outputs:
        loss_hist   : NumPy 1D array of length maxits+1, module loss function value after each iteration
        param_hist  : NumPy 2D matrix of size (maxits+1)-by-(3*num_classes), each row are the  
                      hyperparameters after each iteration
    """
    if scale < 0:
        scale = model._default_scale
    else:
        scale = torch.tensor(scale, dtype=model._dtype, requires_grad=False)

    loss_hist = np.empty(maxits + 1)
    param_hist = np.empty((maxits + 1, model._num_classes * 3))

    if print_info:
        print("Iteration (max %d), Elapsed time (sec), Loss, Hyperparams (before nnt)\n" % maxits)

    t_start = time.time()
    for i in range(maxits):
        # Manually compute loss and set grad
        loss = model.calc_loss_grad(scale)

        # Save history
        loss_hist[i] = loss
        param_hist[i, :] = model._params.detach().numpy()

        # Update parameters
        optimizer.step()

        if print_info:
            t_elapsed = time.time() - t_start
            print("%d, %.2f, %.5f" % (i + 1, t_elapsed, loss))
            print(model._params.detach().numpy(), "")

    # save final loss and parameters
    loss = model.calc_loss_grad(scale)
    loss_hist[maxits] = loss
    param_hist[maxits, :] = model._params.detach().numpy()

    return loss_hist, param_hist

def ezgpr_torch(train_x,
                train_y,
                test_x,
                test_y,
                l_init = 0.0,
                f_init = 0.0,
                s_init = 0.0,
                n_threads = -1,
                exact_gp = 0,
                kernel_type = 1,
                mvtype = 0,
                afn_rank_lq = 50,
                afn_lfil_lq = 0,
                afn_rank_pred = 50,
                afn_lfil_pred = 0,
                niter_lq = 10,
                nvec_lq = 10,
                niter_pred = 500,
                tol_pred = 1e-05,
                dtype_torch = torch.float32,
                seed = 42,
                adam_lr = 0.01,
                adam_maxits = 100):

    """
    Easy to use GP regression interface with PyTorch using Adam optimizer
    Inputs:
        train_x : PyTorch tensor / row-major NumPy array, training data of size d-by-N (or array of size N if d = 1)
        train_y : PyTorch tensor / row-major NumPy array, training labels of size N
        test_x  : PyTorch tensor / row-major NumPy array, testing data of size d-by-N (or array of size N if d = 1)
        test_y  : PyTorch tensor / row-major NumPy array, testing labels of size N
    Optional Inputs (default values):
        l_init (0.0)                      : Initial value of l (before transformation)
        f_init (0.0)                      : Initial value of f (before transformation)
        s_init (0.0)                      : Initial value of s (before transformation)
        n_threads (-1)                    : Number of threads. If negative will use the system's default
        exact_gp (0)                      : Whether to use exact matrix solve in GP computation
        kernel_type (higp.GaussianKernel) : Kernel type, can be higp.GaussianKernel, higp.Matern32Kernel, higp.Matern52Kernel, or higp.CustomKernel.
        mvtype (higp.MatvecAuto)          : Matvec type: can be higp.MatvecAuto, higp.MatvecAOT, or higp.MatvecOTF
        afn_rank_lq (50)                  : The rank of the AFN preconditioner for Lanczos quadrature
        afn_lfil_lq (0)                   : The fill-level of the Schur complement of the AFN preconditioner for Lanczos quadrature
        afn_rank_pred (50)                : The rank of the AFN preconditioner forprediction
        afn_lfil_pred (0)                 : The fill-level of the Schur complement of the AFN preconditioner for prediction
        niter_lq (10)                     : Number of iterations for the Lanczos quadrature
        nvec_lq (10)                      : Number of vectors for the Lanczos quadrature
        niter_pred (500)                  : Number of the PCG solver iterations for the prediction
        tol_pred (1e-5)                   : Prediction PCG solver tolerance
        seed (42)                         : Random seed. If negative will not set seed.
        adam_lr (0.1)                     : Adam optimizer learning rate
        adam_maxits (100)                 : Max number of iterations for the Adam optimizer
        dtype_torch (torch.float32)       : PyTorch datatype
    Outputs:
        pred : structure, containing two members
            pred.prediction_mean   : NumPy array, prediction mean
            pred.prediction_stddev : NumPy array, prediction standard deviation
    """

    if(seed >= 0):
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    dtype = np.float32 if dtype_torch == torch.float32 else np.float64

    if isinstance(train_x, torch.Tensor):
        train_x = np.ascontiguousarray(train_x.numpy().astype(dtype))
    elif isinstance(train_x, np.ndarray):
        train_x = np.ascontiguousarray(train_x.astype(dtype))
    else:
        raise ValueError("Input training data format is not supported")
    if train_x.ndim == 1:
        train_x = train_x[np.newaxis, :]
    
    if isinstance(train_y, torch.Tensor):
        train_y = np.ascontiguousarray(train_y.numpy().astype(dtype))
    elif isinstance(train_y, np.ndarray):
        train_y = np.ascontiguousarray(train_y.astype(dtype))
    else:
        raise ValueError("Input training label format is not supported")
    if train_y.ndim == 2:
        train_y = train_y.squeeze()

    if isinstance(test_x, torch.Tensor):
        test_x = np.ascontiguousarray(test_x.numpy().astype(dtype))
    elif isinstance(test_x, np.ndarray):
        test_x = np.ascontiguousarray(test_x.astype(dtype))
    else:
        raise ValueError("Input testing data format is not supported")
    if test_x.ndim == 1:
        test_x = test_x[np.newaxis, :]

    if isinstance(test_y, torch.Tensor):
        test_y = np.ascontiguousarray(test_y.numpy().astype(dtype))
    elif isinstance(test_y, np.ndarray):
        test_y = np.ascontiguousarray(test_y.astype(dtype))    
    else:
        raise ValueError("Input testing label format is not supported")
    if test_y.ndim == 2:
            test_y = test_y.squeeze()

    print("Read %d training / %d test data points" % (train_x.shape[1], test_x.shape[1]))
    print("Data dimension: %d" % train_x.shape[0])

    N1 = train_x.shape[1]
    N2 = test_x.shape[1]

    gprproblem = higp_cext.gprproblem.setup(data=train_x,
                                            label=train_y,
                                            kernel_type=kernel_type,
                                            exact_gp=exact_gp,
                                            mvtype=mvtype,
                                            nthreads=n_threads,
                                            rank=afn_rank_lq,
                                            lfil=afn_lfil_lq,
                                            niter=niter_lq,
                                            nvec=nvec_lq)

    model = GPRModel(gprproblem, l=l_init, f=f_init, s=s_init, dtype=dtype_torch)
    optimizer = torch.optim.Adam(model.parameters(), lr=adam_lr)

    t0 = time.time()
    gpr_torch_minimize(model, optimizer, maxits=adam_maxits, scale=1.0/N1, print_info=True)
    t1 = time.time()
    print("Training time: %g" % (t1-t0))

    t0 = time.time()
    pred = gpr_prediction(data_train=train_x,
                          label_train=train_y,
                          data_prediction=test_x,
                          kernel_type=kernel_type,
                          exact_gp=exact_gp,
                          mvtype=mvtype,
                          pyparams=model.get_params(),
                          nthreads=n_threads,
                          rank=afn_rank_pred,
                          lfil=afn_lfil_pred,
                          niter=niter_pred,
                          tol=tol_pred)
    t1 = time.time()
    print("Prediction time: %g" % (t1-t0))

    rmse = np.linalg.norm(pred.prediction_mean - test_y) / np.sqrt(float(N2))
    print("RMSE: %g\n" % (rmse))

    return pred
