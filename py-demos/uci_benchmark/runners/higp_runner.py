"""HiGP runner for UCI benchmark."""

import os
import sys
import time
import gc
import ctypes
import tempfile
import stat
from contextlib import contextmanager
import numpy as np
import torch

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

import higp


@contextmanager
def suppress_output():
    """Context manager to suppress stdout/stderr from C extensions."""
    try:
        libc = ctypes.CDLL(None)
        libc.fflush(None)
    except (OSError, AttributeError):
        pass

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
        try:
            libc = ctypes.CDLL(None)
            libc.fflush(None)
        except (OSError, AttributeError):
            pass

        os.dup2(old_stdout_fd, 1)
        os.dup2(old_stderr_fd, 2)
        os.close(old_stdout_fd)
        os.close(old_stderr_fd)

        sys.stdout.flush()
        sys.stderr.flush()


@contextmanager
def capture_output():
    """Capture stdout/stderr and return the text."""
    try:
        libc = ctypes.CDLL(None)
        libc.fflush(None)
    except (OSError, AttributeError):
        pass

    old_stdout_fd = os.dup(1)
    old_stderr_fd = os.dup(2)
    tmp_fd = -1
    tmp_filename = None

    try:
        tmp_fd, tmp_filename = tempfile.mkstemp(prefix="higp_capture_", suffix=".txt")
        os.chmod(tmp_filename, stat.S_IRUSR | stat.S_IWUSR)

        sys.stdout.flush()
        sys.stderr.flush()

        os.dup2(tmp_fd, 1)
        os.dup2(tmp_fd, 2)
        os.close(tmp_fd)
        tmp_fd = -1

        yield tmp_filename

    finally:
        try:
            libc = ctypes.CDLL(None)
            libc.fflush(None)
        except (OSError, AttributeError):
            pass

        os.dup2(old_stdout_fd, 1)
        os.dup2(old_stderr_fd, 2)
        os.close(old_stdout_fd)
        os.close(old_stderr_fd)

        if tmp_fd >= 0:
            try:
                os.close(tmp_fd)
            except OSError:
                pass

        sys.stdout.flush()
        sys.stderr.flush()


def run_higp(
    train_x,
    train_y,
    test_x,
    test_y,
    params,
    dataset_name=None,
    dtype_str="float32",
    seed=42,
):
    """Run HiGP on UCI dataset.

    Args:
        train_x: Training features (D, N) in HiGP format
        train_y: Training labels (N,)
        test_x: Test features (D, N_test) in HiGP format
        test_y: Test labels (N_test,)
        params: Dictionary with HiGP parameters
        dataset_name: Name of dataset for auto-configuration
        dtype_str: Data type string ("float32" or "float64")
        seed: Random seed

    Returns:
        Dictionary with results
    """
    if dtype_str == "float64":
        torch.set_default_dtype(torch.float64)
        np_dtype = np.float64
        torch_dtype = torch.float64
    else:
        torch.set_default_dtype(torch.float32)
        np_dtype = np.float32
        torch_dtype = torch.float32

    torch.manual_seed(seed)
    np.random.seed(seed)

    mvtype = params.get("mvtype", 0)

    train_x = train_x.astype(np_dtype)
    train_y = train_y.astype(np_dtype)
    test_x = test_x.astype(np_dtype)
    test_y = test_y.astype(np_dtype)

    kernel_map = {
        "gaussian": higp.GaussianKernel,
        "matern32": higp.Matern32Kernel,
        "matern52": higp.Matern52Kernel,
    }
    kernel_type = kernel_map.get(
        params.get("kernel", "gaussian").lower(), higp.GaussianKernel
    )

    actual_config = {
        "dtype": "float32" if dtype_str == "float32" else "float64",
        "kernel": params.get("kernel", "gaussian").lower(),
    }

    t_train_start = time.perf_counter()

    with capture_output() as tmp_file:
        gprproblem = higp.gprproblem.setup(
            data=train_x,
            label=train_y,
            kernel_type=kernel_type,
            mvtype=mvtype,
            afn_rank=params.get("train_afn_rank", params.get("afn_rank", 10)),
            afn_lfil=params.get("train_afn_lfil", params.get("afn_lfil", 0)),
            niter=params.get("train_cg_niter", params.get("cg_iters", 20)),
            nvec=params.get("train_cg_nvec", 10),
            seed=seed,
        )

        model = higp.GPRModel(gprproblem, dtype=torch_dtype)

        init_pred = higp.gpr_prediction(
            data_train=train_x,
            label_train=train_y,
            data_prediction=test_x,
            kernel_type=kernel_type,
            gp_params=model.get_params(),
            mvtype=mvtype,
            niter=params.get("pred_cg_niter", params.get("cg_iters", 50)),
            tol=params.get("pred_cg_tol", 1e-2),
            afn_rank=params.get("pred_afn_rank", params.get("afn_rank", 100)),
            afn_lfil=params.get("pred_afn_lfil", params.get("afn_lfil", 0)),
        )

        init_y_pred = init_pred.prediction_mean
        init_y_std = init_pred.prediction_stddev

    try:
        with open(tmp_file, "r") as f:
            output_lines = f.readlines()
        os.unlink(tmp_file)

        for line in output_lines:
            line = line.strip()
            if "Data type:" in line:
                if "float64" in line or "double" in line:
                    actual_config["dtype"] = "float64"
                else:
                    actual_config["dtype"] = "float32"
            elif "Kernel type:" in line:
                if "Gaussian" in line:
                    actual_config["kernel"] = "gaussian"
                elif "Matern32" in line or "Matern 3/2" in line:
                    actual_config["kernel"] = "matern32"
                elif "Matern52" in line or "Matern 5/2" in line:
                    actual_config["kernel"] = "matern52"
            elif "kernel matrix form:" in line:
                if "dense" in line:
                    actual_config["matrix_form"] = "dense/on-the-fly"
                elif "h2" in line or "H2" in line:
                    actual_config["matrix_form"] = "H2"
            elif "AFN preconditioner parameters:" in line:
                import re

                rank_match = re.search(r"rank\s+(\d+)", line)
                lfil_match = re.search(r"lfil\s+(\d+)", line)
                if rank_match:
                    actual_config["afn_rank"] = int(rank_match.group(1))
                if lfil_match:
                    actual_config["afn_lfil"] = int(lfil_match.group(1))
    except (OSError, IOError, ValueError):
        pass

    optimizer = torch.optim.Adam(model.parameters(), lr=params.get("lr", 0.01))

    with suppress_output():
        loss_history, _ = higp.gpr_torch_minimize(
            model, optimizer, maxits=params.get("maxits", 50), print_info=False
        )

    t_train_end = time.perf_counter()

    t_pred_start = time.perf_counter()

    with suppress_output():
        pred = higp.gpr_prediction(
            data_train=train_x,
            label_train=train_y,
            data_prediction=test_x,
            kernel_type=kernel_type,
            gp_params=model.get_params(),
            mvtype=mvtype,
            niter=params.get("pred_cg_niter", params.get("cg_iters", 50)),
            tol=params.get("pred_cg_tol", 1e-2),
            afn_rank=params.get("pred_afn_rank", params.get("afn_rank", 100)),
            afn_lfil=params.get("pred_afn_lfil", params.get("afn_lfil", 0)),
        )

    t_pred_end = time.perf_counter()

    expected_dtype = np_dtype
    assert (
        pred.prediction_mean.dtype == expected_dtype
    ), f"HiGP prediction mean dtype {pred.prediction_mean.dtype} does not match expected {expected_dtype}"
    assert (
        pred.prediction_stddev.dtype == expected_dtype
    ), f"HiGP prediction stddev dtype {pred.prediction_stddev.dtype} does not match expected {expected_dtype}"
    assert (
        init_y_pred.dtype == expected_dtype
    ), f"HiGP initial prediction mean dtype {init_y_pred.dtype} does not match expected {expected_dtype}"
    assert (
        init_y_std.dtype == expected_dtype
    ), f"HiGP initial prediction stddev dtype {init_y_std.dtype} does not match expected {expected_dtype}"

    results = {
        "y_pred": pred.prediction_mean,
        "y_std": pred.prediction_stddev,
        "init_y_pred": init_y_pred,
        "init_y_std": init_y_std,
        "training_time": t_train_end - t_train_start,
        "inference_time": t_pred_end - t_pred_start,
        "loss_history": list(loss_history),
        "init_loss": float(loss_history[0]) if len(loss_history) > 0 else None,
        "final_loss": float(loss_history[-1]) if len(loss_history) > 0 else None,
        "hyperparams": model.get_params().tolist(),
        "dtype": str(np_dtype),
        "actual_config": actual_config,
    }

    with suppress_output():
        del gprproblem
        del model
        gc.collect()

    return results
