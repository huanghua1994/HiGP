#!/usr/bin/env python
import os
import sys
import json
import argparse
import numpy as np
from datetime import datetime
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from uci_benchmark.utils import (
    load_uci_dataset,
    train_test_normalize,
    compute_rmse,
    compute_r2,
    compute_nll,
)
from uci_benchmark.runners import run_higp, run_gpytorch


def run_single_experiment(
    dataset_name,
    n_train,
    n_test,
    higp_params,
    gpytorch_params,
    kernel,
    dtype_str,
    seed,
    repeat_id,
    skip_gpytorch=False,
):
    """Run a single experiment comparing HiGP and GPyTorch"""

    print(f"  Loading {dataset_name} dataset...")

    train_x, train_y, test_x, test_y = load_uci_dataset(
        dataset_name,
        n_train=n_train,
        n_test=n_test,
        seed=seed + repeat_id * 1000,
    )

    train_x, train_y, test_x, test_y = train_test_normalize(
        train_x, train_y, test_x, test_y
    )

    print(f"    Data loaded: {n_train} train, {n_test} test samples")
    print(f"    Features: {train_x.shape[1]}")

    print(f"  Running HiGP...")
    higp_params_with_kernel = {**higp_params, "kernel": kernel}
    higp_results = run_higp(
        train_x,
        train_y,
        test_x,
        test_y,
        higp_params_with_kernel,
        dtype_str=dtype_str,
        seed=seed,
    )

    higp_init_rmse = compute_rmse(test_y, higp_results["init_y_pred"])
    higp_init_r2 = compute_r2(test_y, higp_results["init_y_pred"])
    higp_init_nll = compute_nll(
        test_y, higp_results["init_y_pred"], higp_results["init_y_std"]
    )

    print(f"    Initial RMSE: {higp_init_rmse:.6f}")

    higp_rmse = compute_rmse(test_y, higp_results["y_pred"])
    higp_r2 = compute_r2(test_y, higp_results["y_pred"])
    higp_nll = compute_nll(test_y, higp_results["y_pred"], higp_results["y_std"])

    print(f"    Final RMSE: {higp_rmse:.6f}")
    print(f"    Training time: {higp_results['training_time']:.2f}s")

    if not skip_gpytorch:
        print(f"  Running GPyTorch...")
        gpytorch_params_with_kernel = {**gpytorch_params, "kernel": kernel}
        gpytorch_results = run_gpytorch(
            train_x,
            train_y,
            test_x,
            test_y,
            gpytorch_params_with_kernel,
            dtype_str=dtype_str,
            seed=seed,
        )

        gpytorch_init_rmse = compute_rmse(test_y, gpytorch_results["init_y_pred"])
        gpytorch_init_r2 = compute_r2(test_y, gpytorch_results["init_y_pred"])
        gpytorch_init_nll = compute_nll(
            test_y, gpytorch_results["init_y_pred"], gpytorch_results["init_y_std"]
        )

        print(f"    Initial RMSE: {gpytorch_init_rmse:.6f}")

        gpytorch_rmse = compute_rmse(test_y, gpytorch_results["y_pred"])
        gpytorch_r2 = compute_r2(test_y, gpytorch_results["y_pred"])
        gpytorch_nll = compute_nll(
            test_y, gpytorch_results["y_pred"], gpytorch_results["y_std"]
        )

        print(f"    Final RMSE: {gpytorch_rmse:.6f}")
        print(f"    Training time: {gpytorch_results['training_time']:.2f}s")
    else:
        gpytorch_results = None
        gpytorch_init_rmse = None
        gpytorch_init_r2 = None
        gpytorch_init_nll = None
        gpytorch_rmse = None
        gpytorch_r2 = None
        gpytorch_nll = None

    result = {
        "dataset": dataset_name,
        "n_train": n_train,
        "n_test": n_test,
        "n_features": train_x.shape[1],
        "kernel": kernel,
        "repeat_id": repeat_id,
        "higp": {
            "init_rmse": higp_init_rmse,
            "init_r2": higp_init_r2,
            "init_nll": higp_init_nll,
            "rmse": higp_rmse,
            "r2": higp_r2,
            "nll": higp_nll,
            "training_time": higp_results["training_time"],
            "inference_time": higp_results["inference_time"],
            "init_loss": higp_results["init_loss"],
            "final_loss": higp_results["final_loss"],
            "dtype": higp_results.get("dtype", "unknown"),
            "actual_config": higp_results.get("actual_config", {}),
        },
    }

    if not skip_gpytorch:
        result["gpytorch"] = {
            "init_rmse": gpytorch_init_rmse,
            "init_r2": gpytorch_init_r2,
            "init_nll": gpytorch_init_nll,
            "rmse": gpytorch_rmse,
            "r2": gpytorch_r2,
            "nll": gpytorch_nll,
            "training_time": gpytorch_results["training_time"],
            "inference_time": gpytorch_results["inference_time"],
            "init_loss": gpytorch_results["init_loss"],
            "final_loss": gpytorch_results["final_loss"],
            "cg_warning": gpytorch_results.get("cg_warning", False),
        }

    summary_result = {
        "dataset": dataset_name,
        "n_train": n_train,
        "n_test": n_test,
        "n_features": train_x.shape[1],
        "kernel": kernel,
        "repeat_id": repeat_id,
        "higp": {
            "init_rmse": higp_init_rmse,
            "init_r2": higp_init_r2,
            "init_nll": higp_init_nll,
            "rmse": higp_rmse,
            "r2": higp_r2,
            "nll": higp_nll,
            "training_time": higp_results["training_time"],
            "inference_time": higp_results["inference_time"],
        },
    }

    if not skip_gpytorch:
        summary_result["gpytorch"] = {
            "init_rmse": gpytorch_init_rmse,
            "init_r2": gpytorch_init_r2,
            "init_nll": gpytorch_init_nll,
            "rmse": gpytorch_rmse,
            "r2": gpytorch_r2,
            "nll": gpytorch_nll,
            "training_time": gpytorch_results["training_time"],
            "inference_time": gpytorch_results["inference_time"],
            "cg_warning": gpytorch_results.get("cg_warning", False),
        }

    detailed = result.copy()
    detailed["higp"]["y_pred"] = higp_results["y_pred"].tolist()
    detailed["higp"]["y_std"] = higp_results["y_std"].tolist()
    detailed["higp"]["init_y_pred"] = higp_results["init_y_pred"].tolist()
    detailed["higp"]["init_y_std"] = higp_results["init_y_std"].tolist()
    detailed["higp"]["loss_history"] = higp_results["loss_history"]
    detailed["higp"]["hyperparams"] = higp_results["hyperparams"]
    detailed["test_y"] = test_y.tolist()

    if not skip_gpytorch and gpytorch_results:
        detailed["gpytorch"]["y_pred"] = gpytorch_results["y_pred"].tolist()
        detailed["gpytorch"]["y_std"] = gpytorch_results["y_std"].tolist()
        detailed["gpytorch"]["init_y_pred"] = gpytorch_results["init_y_pred"].tolist()
        detailed["gpytorch"]["init_y_std"] = gpytorch_results["init_y_std"].tolist()
        detailed["gpytorch"]["loss_history"] = gpytorch_results["loss_history"]
        detailed["gpytorch"]["hyperparams"] = gpytorch_results["hyperparams"]

    return summary_result, detailed


def main():
    parser = argparse.ArgumentParser(description="Run UCI benchmark")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to config JSON file"
    )
    parser.add_argument(
        "--output_dir", type=str, default="results", help="Output directory"
    )
    parser.add_argument(
        "--higp-only", action="store_true", help="Only run HiGP experiments"
    )

    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = json.load(f)

    datasets = config.get("datasets", ["bike", "road3d"])
    n_repeats = config.get("n_repeats", 3)
    kernel = config.get("kernel", "gaussian")
    seed = config.get("seed", 42)
    higp_params = config.get("higp_params", {})
    gpytorch_params = config.get("gpytorch_params", {})
    dtype_str = config.get("dtype", "float32")

    if "n_train_configs" in config:
        n_train_configs = config["n_train_configs"]
    else:
        n_train = config.get("n_train", 1000)
        n_train_configs = {dataset: [n_train] for dataset in datasets}

    n_test = config.get("n_test", 200)

    os.makedirs(args.output_dir, exist_ok=True)

    total_experiments = sum(
        len(n_train_configs.get(dataset, [config.get("n_train", 1000)])) * n_repeats
        for dataset in datasets
    )

    print(f"Running {total_experiments} experiments from config: {args.config}")
    print(f"Datasets: {datasets}")
    print(f"Training sizes: {n_train_configs}")
    print(f"Test size: {n_test}")
    print(f"Kernel: {kernel}")
    print(f"Repeats per config: {n_repeats}")

    if args.higp_only:
        print("Mode: HiGP only")
    else:
        print("Mode: Comparing HiGP and GPyTorch")

    all_results = []
    all_detailed = []
    exp_count = 0

    for dataset in datasets:
        train_sizes = n_train_configs.get(dataset, [config.get("n_train", 1000)])

        for n_train in train_sizes:
            for repeat_id in range(n_repeats):
                exp_count += 1
                print(
                    f"\nExperiment {exp_count}/{total_experiments}: "
                    f"{dataset} n_train={n_train} repeat={repeat_id+1}"
                )

                result, detailed = run_single_experiment(
                    dataset_name=dataset,
                    n_train=n_train,
                    n_test=n_test,
                    higp_params=higp_params,
                    gpytorch_params=gpytorch_params,
                    kernel=kernel,
                    dtype_str=dtype_str,
                    seed=seed,
                    repeat_id=repeat_id,
                    skip_gpytorch=args.higp_only,
                )

                all_results.append(result)
                all_detailed.append(detailed)

    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)

    grouped = defaultdict(list)
    for result in all_results:
        key = (result["dataset"], result["n_train"])
        grouped[key].append(result)

    for (dataset, n_train), group_results in sorted(grouped.items()):
        print(f"\n{dataset} (n_train={n_train}):")

        higp_init_rmses = [r["higp"]["init_rmse"] for r in group_results]
        higp_rmses = [r["higp"]["rmse"] for r in group_results]
        higp_times = [r["higp"]["training_time"] for r in group_results]

        print(f"  HiGP:")
        print(
            f"    Initial RMSE: {np.mean(higp_init_rmses):.6f} ± {np.std(higp_init_rmses):.6f}"
        )
        print(f"    Final RMSE: {np.mean(higp_rmses):.6f} ± {np.std(higp_rmses):.6f}")
        print(
            f"    Training time: {np.mean(higp_times):.2f}s ± {np.std(higp_times):.2f}s"
        )

        if not args.higp_only and "gpytorch" in group_results[0]:
            gpytorch_init_rmses = [r["gpytorch"]["init_rmse"] for r in group_results]
            gpytorch_rmses = [r["gpytorch"]["rmse"] for r in group_results]
            gpytorch_times = [r["gpytorch"]["training_time"] for r in group_results]

            print(f"  GPyTorch:")
            print(
                f"    Initial RMSE: {np.mean(gpytorch_init_rmses):.6f} ± {np.std(gpytorch_init_rmses):.6f}"
            )
            print(
                f"    Final RMSE: {np.mean(gpytorch_rmses):.6f} ± {np.std(gpytorch_rmses):.6f}"
            )
            print(
                f"    Training time: {np.mean(gpytorch_times):.2f}s ± {np.std(gpytorch_times):.2f}s"
            )

    if all_results:
        for r in all_results:
            if "higp" in r and "actual_config" in r["higp"]:
                actual_config = r["higp"]["actual_config"]
                if actual_config:
                    print("\nActual Configuration Used:")
                    if "dtype" in actual_config:
                        print(f"  Data type: {actual_config['dtype']}")
                    if "kernel" in actual_config:
                        print(f"  Kernel: {actual_config['kernel']}")
                    if "matrix_form" in actual_config:
                        print(f"  Matrix form: {actual_config['matrix_form']}")
                    break

    if any(r.get("gpytorch", {}).get("cg_warning", False) for r in all_results):
        print("\nCG convergence warnings detected during GPyTorch training.")
        print("Consider increasing train_cg_niter if accuracy is affected.")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config_name = config.get("name", "uci_benchmark")

    summary_file = os.path.join(
        args.output_dir, f"{config_name}_summary_{timestamp}.json"
    )
    detailed_file = os.path.join(
        args.output_dir, f"{config_name}_detailed_{timestamp}.json"
    )

    with open(summary_file, "w") as f:
        json.dump({"config": config, "results": all_results}, f, indent=2)

    with open(detailed_file, "w") as f:
        json.dump({"config": config, "results": all_detailed}, f, indent=2)

    print(f"\nResults saved to:")
    print(f"  Summary: {summary_file}")
    print(f"  Detailed: {detailed_file}")


if __name__ == "__main__":
    main()
