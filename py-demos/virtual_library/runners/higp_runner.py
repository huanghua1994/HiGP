import time
import numpy as np
import torch
import higp
import os
import sys
import gc
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

def run_higp(train_x, train_y, test_x, test_y, params, seed=42):
    """Run HiGP on the given data

    Args:
        train_x: Training features (N, D)
        train_y: Training targets (N,)
        test_x: Test features (M, D)
        test_y: Test targets (M,)
        params: Dictionary with HiGP parameters
        seed: Random seed

    Returns:
        Dictionary with results
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Convert to HiGP format: (D, N)
    train_x_higp = np.ascontiguousarray(train_x.T).astype(np.float32)
    train_y_higp = np.ascontiguousarray(train_y).astype(np.float32)
    test_x_higp = np.ascontiguousarray(test_x.T).astype(np.float32)

    with suppress_output():
        gprproblem = higp.gprproblem.setup(
            data=train_x_higp,
            label=train_y_higp,
            kernel_type=higp.GaussianKernel,
            mvtype=params.get("mvtype", 0),
            niter=params.get("train_cg_niter", params.get("cg_iters", 20)),
            nvec=params.get("train_cg_nvec", 10), # same as gpytorch 
            afn_rank=params.get("train_afn_rank", params.get("afn_rank", 5)),
            afn_lfil=params.get("train_afn_lfil", params.get("afn_lfil", 5)),
            seed=seed,
        )

    model = higp.GPRModel(gprproblem, dtype=torch.float32)
    with suppress_output():
        init_pred = higp.gpr_prediction(
            data_train=train_x_higp,
            label_train=train_y_higp,
            data_prediction=test_x_higp,
            kernel_type=higp.GaussianKernel,
            gp_params=model.get_params(),
            mvtype=params.get("mvtype", 0),
            niter=params.get("pred_cg_niter", params.get("cg_iters", 50)),
            tol=params.get("pred_cg_tol", 1e-2),
            afn_rank=params.get("pred_afn_rank", params.get("afn_rank", 50)),
            afn_lfil=params.get("pred_afn_lfil", params.get("afn_lfil", 50)),
        )
    init_y_pred = init_pred.prediction_mean
    init_y_std = init_pred.prediction_stddev
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
            kernel_type=higp.GaussianKernel,
            gp_params=model.get_params(),
            mvtype=params.get("mvtype", 0),
            niter=params.get("pred_cg_niter", params.get("cg_iters", 50)),
            tol=params.get("pred_cg_tol", 1e-6),
            afn_rank=params.get("pred_afn_rank", params.get("afn_rank", 50)),
            afn_lfil=params.get("pred_afn_lfil", params.get("afn_lfil", 50)),
        )

    t_pred_end = time.perf_counter()

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
    }

    # Cleanup C objects with output suppressed
    with suppress_output():
        del gprproblem
        del model
        gc.collect()  # Ensures C destructors run while output is suppressed

    return results
