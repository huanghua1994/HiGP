import time
import numpy as np
import torch
import gpytorch
import os
import sys
from contextlib import contextmanager

@contextmanager
def suppress_output():
    old_stdout_fd = os.dup(1)
    old_stderr_fd = os.dup(2)
    try:
        sys.stdout.flush()
        sys.stderr.flush()
        devnull_fd = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull_fd, 1)
        os.dup2(devnull_fd, 2)
        os.close(devnull_fd)
        yield
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(old_stdout_fd, 1)
        os.dup2(old_stderr_fd, 2)
        os.close(old_stdout_fd)
        os.close(old_stderr_fd)

class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def run_gpytorch(train_x, train_y, test_x, test_y, params, seed=42):
    """Run GPyTorch on the given data

    Args:
        train_x: Training features (N, D)
        train_y: Training targets (N,)
        test_x: Test features (M, D)
        test_y: Test targets (M,)
        params: Dictionary with GPyTorch parameters
        seed: Random seed

    Returns:
        Dictionary with results
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_x_torch = torch.from_numpy(train_x).float()
    train_y_torch = torch.from_numpy(train_y).float()
    test_x_torch = torch.from_numpy(test_x).float()

    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = ExactGPModel(train_x_torch, train_y_torch, likelihood)

    model.eval()
    likelihood.eval()
    
    # Note: gpytorch uses relative tolerance 1e-02, and total number of iterations is default to 1000
    # we won't run to 1000 iterations, see https://docs.gpytorch.ai/en/stable/settings.html#:~:text=class%20gpytorch.settings.eval_cg_tolerance(value)
    # this is the most reasonable setting for comparison with HiGP
    with suppress_output(), torch.no_grad(), gpytorch.settings.fast_pred_var(), gpytorch.settings.max_cg_iterations(
        params.get("pred_cg_niter", params.get("cg_iters", 50))
    ), gpytorch.settings.max_lanczos_quadrature_iterations(
        params.get("pred_cg_niter", params.get("cg_iters", 50))
    ), gpytorch.settings.max_preconditioner_size(
        params.get("pred_max_preconditioner_size", params.get("max_preconditioner_size", 100))
    ):
        init_predictions = likelihood(model(test_x_torch))
        init_y_pred = init_predictions.mean.numpy()
        init_y_var = init_predictions.variance.numpy()
        init_y_std = np.sqrt(init_y_var)

    model.train()
    likelihood.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=params.get("lr", 0.01))
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    loss_history = []

    t_train_start = time.perf_counter()

    # Note: gpytorch uses relative tolerance 1, and total number of iterations is default to 1000
    # we won't run to 1000 iterations, see https://docs.gpytorch.ai/en/stable/settings.html#:~:text=class%20gpytorch.settings.cg_tolerance(value)
    # this is the most reasonable setting for comparison with HiGP
    with suppress_output(), gpytorch.settings.max_cg_iterations(
        params.get("train_cg_niter", params.get("cg_iters", 1000))
    ), gpytorch.settings.max_lanczos_quadrature_iterations(
        params.get("train_cg_niter", params.get("cg_iters", 1000))
    ), gpytorch.settings.max_preconditioner_size(
        params.get("train_max_preconditioner_size", params.get("max_preconditioner_size", 10))
    ):

        with torch.no_grad():
            output0 = model(train_x_torch)
            init_loss = -mll(output0, train_y_torch).item()
            loss_history.append(init_loss)

        for i in range(params.get("maxits", 50)):
            optimizer.zero_grad()
            output = model(train_x_torch)
            loss = -mll(output, train_y_torch)
            loss.backward()
            loss_history.append(loss.item())
            optimizer.step()

    t_train_end = time.perf_counter()

    t_pred_start = time.perf_counter()
    model.eval()
    likelihood.eval()

    # Note: gpytorch uses relative tolerance 1e-02, and total number of iterations is default to 1000
    # we won't run to 1000 iterations, see https://docs.gpytorch.ai/en/stable/settings.html#:~:text=class%20gpytorch.settings.eval_cg_tolerance(value)
    # this is the most reasonable setting for comparison with HiGP
    with suppress_output(), torch.no_grad(), gpytorch.settings.fast_pred_var(), gpytorch.settings.max_cg_iterations(
        params.get("pred_cg_niter", params.get("cg_iters", 1000))
    ), gpytorch.settings.max_lanczos_quadrature_iterations(
        params.get("pred_cg_niter", params.get("cg_iters", 1000))
    ), gpytorch.settings.max_preconditioner_size(
        params.get("pred_max_preconditioner_size", params.get("max_preconditioner_size", 100))
    ):
        predictions = likelihood(model(test_x_torch))
        y_pred = predictions.mean.numpy()
        y_var = predictions.variance.numpy()
        y_std = np.sqrt(y_var)

    t_pred_end = time.perf_counter()

    hyperparams = [
        model.likelihood.noise.item(),
        model.covar_module.base_kernel.lengthscale.detach().numpy().flatten().tolist(),
        model.covar_module.outputscale.item(),
        model.mean_module.constant.item(),
    ]

    results = {
        "y_pred": y_pred,
        "y_std": y_std,
        "init_y_pred": init_y_pred,
        "init_y_std": init_y_std,
        "training_time": t_train_end - t_train_start,
        "inference_time": t_pred_end - t_pred_start,
        "loss_history": loss_history,
        "init_loss": loss_history[0] if len(loss_history) > 0 else None,
        "final_loss": loss_history[-1] if len(loss_history) > 0 else None,
        "hyperparams": hyperparams,
    }

    return results
