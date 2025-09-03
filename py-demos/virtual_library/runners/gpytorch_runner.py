import time
import warnings
import numpy as np
import torch
import gpytorch
from ..utils import suppress_output


class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, kernel_type="gaussian"):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
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
    train_x, train_y, test_x, test_y, params, dtype_str="float32", seed=42
):
    """Run GPyTorch on the given data

    Args:
        train_x: Training features (N, D)
        train_y: Training targets (N,)
        test_x: Test features (M, D)
        test_y: Test targets (M,)
        params: Dictionary with GPyTorch parameters
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

    train_x_torch = torch.from_numpy(train_x).to(torch_dtype)
    train_y_torch = torch.from_numpy(train_y).to(torch_dtype)
    test_x_torch = torch.from_numpy(test_x).to(torch_dtype)

    kernel_type = params.get("kernel", "gaussian").lower()

    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = ExactGPModel(train_x_torch, train_y_torch, likelihood, kernel_type)

    model.eval()
    likelihood.eval()

    # Initial predictions with default GPyTorch CG settings
    with suppress_output(), torch.no_grad(), gpytorch.settings.fast_pred_var(), gpytorch.settings.max_cg_iterations(
        params.get("pred_cg_niter", params.get("cg_iters", 1000))
    ), gpytorch.settings.max_lanczos_quadrature_iterations(
        params.get("pred_cg_niter", params.get("cg_iters", 1000))
    ), gpytorch.settings.max_preconditioner_size(
        params.get(
            "pred_max_preconditioner_size", params.get("max_preconditioner_size", 100)
        )
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
    cg_warning_occurred = False

    t_train_start = time.perf_counter()

    # Training with CG settings
    with suppress_output(), warnings.catch_warnings(record=True) as w:
        # Filter to only catch NumericalWarning from linear_operator
        warnings.filterwarnings(
            "always", category=UserWarning, module="linear_operator"
        )

        with gpytorch.settings.max_cg_iterations(
            params.get("train_cg_niter", params.get("cg_iters", 1000))
        ), gpytorch.settings.max_lanczos_quadrature_iterations(
            params.get("train_cg_niter", params.get("cg_iters", 1000))
        ), gpytorch.settings.max_preconditioner_size(
            params.get(
                "train_max_preconditioner_size",
                params.get("max_preconditioner_size", 10),
            )
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

        if w:
            for warning in w:
                if "CG terminated" in str(warning.message):
                    cg_warning_occurred = True
                    break

    t_train_end = time.perf_counter()

    t_pred_start = time.perf_counter()
    model.eval()
    likelihood.eval()

    # Final predictions with trained model
    with suppress_output(), torch.no_grad(), gpytorch.settings.fast_pred_var(), gpytorch.settings.max_cg_iterations(
        params.get("pred_cg_niter", params.get("cg_iters", 1000))
    ), gpytorch.settings.max_lanczos_quadrature_iterations(
        params.get("pred_cg_niter", params.get("cg_iters", 1000))
    ), gpytorch.settings.max_preconditioner_size(
        params.get(
            "pred_max_preconditioner_size", params.get("max_preconditioner_size", 100)
        )
    ):
        predictions = likelihood(model(test_x_torch))
        y_pred = predictions.mean.numpy()
        y_var = predictions.variance.numpy()
        y_std = np.sqrt(y_var)

    t_pred_end = time.perf_counter()

    # Verify dtype precision matches expected configuration
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
        "cg_warning": cg_warning_occurred,
    }

    return results
