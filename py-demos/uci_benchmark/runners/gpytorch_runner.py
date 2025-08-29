"""GPyTorch runner for UCI benchmark."""

import time
import warnings
import os
import sys
import numpy as np
import torch
import gpytorch
from contextlib import contextmanager


@contextmanager
def suppress_output():
    """Context manager to suppress stdout/stderr from C extensions."""
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
    """Standard ExactGP model for UCI datasets."""

    def __init__(self, train_x, train_y, likelihood, kernel_type="gaussian"):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()

        if kernel_type == "matern32":
            base_kernel = gpytorch.kernels.MaternKernel(nu=1.5)
        elif kernel_type == "matern52":
            base_kernel = gpytorch.kernels.MaternKernel(nu=2.5)
        else:
            base_kernel = gpytorch.kernels.RBFKernel()

        self.covar_module = gpytorch.kernels.ScaleKernel(base_kernel)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def run_gpytorch(
    train_x,
    train_y,
    test_x,
    test_y,
    params,
    dataset_name=None,
    dtype_str="float32",
    seed=42,
):
    """Run GPyTorch on UCI dataset.

    Args:
        train_x: Training features (D, N) in HiGP format
        train_y: Training labels (N,)
        test_x: Test features (D, N_test) in HiGP format
        test_y: Test labels (N_test,)
        params: Dictionary with GPyTorch parameters
        dataset_name: Name of dataset for auto-configuration
        dtype_str: Data type string ("float32" or "float64")
        seed: Random seed

    Returns:
        Dictionary with results
    """
    if dtype_str == "float64":
        torch.set_default_dtype(torch.float64)
        torch_dtype = torch.float64
    else:
        torch.set_default_dtype(torch.float32)
        torch_dtype = torch.float32

    torch.manual_seed(seed)
    np.random.seed(seed)

    train_x_gp = torch.tensor(train_x.T, dtype=torch_dtype)
    train_y_gp = torch.tensor(train_y, dtype=torch_dtype)
    test_x_gp = torch.tensor(test_x.T, dtype=torch_dtype)
    test_y_gp = torch.tensor(test_y, dtype=torch_dtype)

    t_train_start = time.perf_counter()

    kernel_type = params.get("kernel", "gaussian").lower()

    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = ExactGPModel(train_x_gp, train_y_gp, likelihood, kernel_type)

    model.eval()
    likelihood.eval()
    with torch.no_grad(), warnings.catch_warnings(), gpytorch.settings.fast_pred_var(), gpytorch.settings.max_preconditioner_size(
        params.get("pred_max_preconditioner_size", 100)
    ), gpytorch.settings.max_cg_iterations(
        params.get("pred_cg_niter", 1000)
    ), gpytorch.settings.max_lanczos_quadrature_iterations(
        params.get("pred_cg_niter", 1000)
    ):
        warnings.simplefilter("ignore")
        init_output = model(test_x_gp)
        init_preds = likelihood(init_output)
        init_y_pred = init_preds.mean.numpy()
        init_y_std = init_preds.stddev.numpy()

    model.train()
    likelihood.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=params.get("lr", 0.01))

    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    cg_iters = params.get("train_cg_niter", params.get("cg_iters", 1000))
    max_preconditioner_size = params.get(
        "train_max_preconditioner_size", params.get("max_preconditioner_size", 10)
    )

    loss_history = []
    cg_warning_occurred = False

    with warnings.catch_warnings(record=True) as w:
        warnings.filterwarnings(
            "always", category=UserWarning, module="linear_operator"
        )

        with gpytorch.settings.max_preconditioner_size(
            max_preconditioner_size
        ), gpytorch.settings.max_cg_iterations(
            cg_iters
        ), gpytorch.settings.max_lanczos_quadrature_iterations(
            cg_iters
        ):

            with torch.no_grad():
                output0 = model(train_x_gp)
                init_loss = -mll(output0, train_y_gp).item()
                loss_history.append(init_loss)

            for i in range(params.get("maxits", 50)):
                optimizer.zero_grad()
                output = model(train_x_gp)
                loss = -mll(output, train_y_gp)
                loss.backward()
                optimizer.step()
                loss_history.append(loss.item())

        if w:
            for warning in w:
                if "CG terminated" in str(warning.message):
                    cg_warning_occurred = True
                    break

    t_train_end = time.perf_counter()

    t_pred_start = time.perf_counter()

    model.eval()
    likelihood.eval()

    pred_cg_iters = params.get("pred_cg_niter", params.get("cg_iters", 1000))
    pred_preconditioner_size = params.get(
        "pred_max_preconditioner_size", params.get("max_preconditioner_size", 100)
    )

    with torch.no_grad(), warnings.catch_warnings(), gpytorch.settings.fast_pred_var(), gpytorch.settings.max_preconditioner_size(
        pred_preconditioner_size
    ), gpytorch.settings.max_cg_iterations(
        pred_cg_iters
    ), gpytorch.settings.max_lanczos_quadrature_iterations(
        pred_cg_iters
    ):
        warnings.simplefilter("ignore")
        test_output = model(test_x_gp)
        observed_pred = likelihood(test_output)
        y_pred = observed_pred.mean.numpy()
        y_std = observed_pred.stddev.numpy()

    t_pred_end = time.perf_counter()

    expected_dtype = np.float64 if torch_dtype == torch.float64 else np.float32
    assert (
        y_pred.dtype == expected_dtype
    ), f"GPyTorch prediction mean dtype {y_pred.dtype} does not match expected {expected_dtype}"
    assert (
        y_std.dtype == expected_dtype
    ), f"GPyTorch prediction stddev dtype {y_std.dtype} does not match expected {expected_dtype}"
    assert (
        init_y_pred.dtype == expected_dtype
    ), f"GPyTorch initial prediction mean dtype {init_y_pred.dtype} does not match expected {expected_dtype}"
    assert (
        init_y_std.dtype == expected_dtype
    ), f"GPyTorch initial prediction stddev dtype {init_y_std.dtype} does not match expected {expected_dtype}"

    with torch.no_grad():
        lengthscale = model.covar_module.base_kernel.lengthscale.item()
        outputscale = model.covar_module.outputscale.item()
        noise = likelihood.noise.item()
        hyperparams = [lengthscale, outputscale, noise]

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
        "cg_warning": cg_warning_occurred,
    }

    return results
