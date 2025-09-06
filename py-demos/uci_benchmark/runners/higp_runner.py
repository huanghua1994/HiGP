import time
import numpy as np
import torch
import higp
import gc
import os
from ..utils import suppress_output, capture_output, prepare_for_higp


def run_higp(train_x, train_y, test_x, test_y, params, dtype_str="float32", seed=42):
    """Run HiGP on the given data"""
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

    train_x_higp, train_y_higp = prepare_for_higp(train_x, train_y, dtype=np_dtype)
    test_x_higp, _ = prepare_for_higp(test_x, np.array([]), dtype=np_dtype)

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

    with capture_output() as tmp_file:
        gprproblem = higp.gprproblem.setup(
            data=train_x_higp,
            label=train_y_higp,
            kernel_type=kernel_type,
            mvtype=params.get("mvtype", 0),
            afn_rank=params.get("train_afn_rank", params.get("afn_rank", 10)),
            afn_lfil=params.get("train_afn_lfil", params.get("afn_lfil", 0)),
            niter=params.get("train_cg_niter", params.get("cg_iters", 20)),
            nvec=params.get("train_cg_nvec", 10),
            seed=seed,
        )

        model = higp.GPRModel(gprproblem, dtype=torch_dtype)

        init_pred = higp.gpr_prediction(
            data_train=train_x_higp,
            label_train=train_y_higp,
            data_prediction=test_x_higp,
            kernel_type=kernel_type,
            gp_params=model.get_params(),
            mvtype=params.get("mvtype", 0),
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

    t_train_start = time.perf_counter()
    with suppress_output():
        loss_history, _ = higp.gpr_torch_minimize(
            model, optimizer, maxits=params.get("maxits", 50), print_info=False
        )
    t_train_end = time.perf_counter()

    t_pred_start = time.perf_counter()
    with suppress_output():
        pred = higp.gpr_prediction(
            data_train=train_x_higp,
            label_train=train_y_higp,
            data_prediction=test_x_higp,
            kernel_type=kernel_type,
            gp_params=model.get_params(),
            mvtype=params.get("mvtype", 0),
            niter=params.get("pred_cg_niter", params.get("cg_iters", 50)),
            tol=params.get("pred_cg_tol", 1e-2),
            afn_rank=params.get("pred_afn_rank", params.get("afn_rank", 100)),
            afn_lfil=params.get("pred_afn_lfil", params.get("afn_lfil", 0)),
        )

    t_pred_end = time.perf_counter()

    expected_dtype = np_dtype
    assert (
        pred.prediction_mean.dtype == expected_dtype
    ), f"{pred.prediction_mean.dtype} != {expected_dtype}"
    assert (
        pred.prediction_stddev.dtype == expected_dtype
    ), f"{pred.prediction_stddev.dtype} != {expected_dtype}"
    assert (
        init_y_pred.dtype == expected_dtype
    ), f"{init_y_pred.dtype} != {expected_dtype}"
    assert init_y_std.dtype == expected_dtype, f"{init_y_std.dtype} != {expected_dtype}"

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
