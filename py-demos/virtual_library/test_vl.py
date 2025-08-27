#!/usr/bin/env python
import os
import sys
import json
import argparse
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from virtual_library.utils import (
    get_function_info,
    generate_training_data,
    compute_rmse,
    compute_r2,
    compute_nll,
)
from virtual_library.runners import run_higp, run_gpytorch


def run_single_experiment(
    func_name,
    dim,
    n_train,
    n_test,
    noise_level,
    higp_params,
    gpytorch_params,
    seed,
    repeat_id,
    skip_gpytorch=False,
):
    """Run a single experiment comparing HiGP and GPyTorch"""

    func_info = get_function_info(func_name)
    if func_info is None:
        raise ValueError(f"Unknown function: {func_name}")

    if func_info["supports_dims"] != "any" and dim != func_info["supports_dims"]:
        print(
            f"Skipping {func_name} for dim={dim} (only supports {func_info['supports_dims']})"
        )
        return None

    # Data generation
    func = func_info["func"]
    bounds = func_info["bounds"]
    data_seed = seed + repeat_id * 1000
    train_x, train_y, test_x, test_y = generate_training_data(
        func, n_train, n_test, dim, bounds, noise_level=noise_level, seed=data_seed
    )

    # Run HiGP
    print(f"  Running HiGP...")
    higp_results = run_higp(train_x, train_y, test_x, test_y, higp_params, seed=seed)
    higp_init_rmse = compute_rmse(test_y, higp_results["init_y_pred"])
    higp_init_r2 = compute_r2(test_y, higp_results["init_y_pred"])
    higp_init_nll = compute_nll(
        test_y, higp_results["init_y_pred"], higp_results["init_y_std"]
    )
    higp_rmse = compute_rmse(test_y, higp_results["y_pred"])
    higp_r2 = compute_r2(test_y, higp_results["y_pred"])
    higp_nll = compute_nll(test_y, higp_results["y_pred"], higp_results["y_std"])

    # Run GPyTorch (if not skipped)
    if not skip_gpytorch:
        print(f"  Running GPyTorch...")
        gpytorch_results = run_gpytorch(
            train_x, train_y, test_x, test_y, gpytorch_params, seed=seed
        )
        gpytorch_init_rmse = compute_rmse(test_y, gpytorch_results["init_y_pred"])
        gpytorch_init_r2 = compute_r2(test_y, gpytorch_results["init_y_pred"])
        gpytorch_init_nll = compute_nll(
            test_y, gpytorch_results["init_y_pred"], gpytorch_results["init_y_std"]
        )
        gpytorch_rmse = compute_rmse(test_y, gpytorch_results["y_pred"])
        gpytorch_r2 = compute_r2(test_y, gpytorch_results["y_pred"])
        gpytorch_nll = compute_nll(
            test_y, gpytorch_results["y_pred"], gpytorch_results["y_std"]
        )
    else:
        gpytorch_results = None
        gpytorch_init_rmse = None
        gpytorch_init_r2 = None
        gpytorch_init_nll = None
        gpytorch_rmse = None
        gpytorch_r2 = None
        gpytorch_nll = None

    # Save results
    result = {
        "function": func_name,
        "dim": dim,
        "n_train": n_train,
        "n_test": n_test,
        "noise_level": noise_level,
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
        }

    # Prepare detailed results with all data
    detailed_result = result.copy()
    detailed_result["higp"]["loss_history"] = higp_results["loss_history"]
    detailed_result["higp"]["predictions"] = higp_results["y_pred"].tolist()
    detailed_result["higp"]["std_predictions"] = higp_results["y_std"].tolist()
    detailed_result["higp"]["init_predictions"] = higp_results["init_y_pred"].tolist()
    detailed_result["higp"]["init_std_predictions"] = higp_results[
        "init_y_std"
    ].tolist()
    detailed_result["higp"]["hyperparams"] = higp_results["hyperparams"]

    if not skip_gpytorch:
        detailed_result["gpytorch"]["loss_history"] = gpytorch_results["loss_history"]
        detailed_result["gpytorch"]["predictions"] = gpytorch_results["y_pred"].tolist()
        detailed_result["gpytorch"]["std_predictions"] = gpytorch_results[
            "y_std"
        ].tolist()
        detailed_result["gpytorch"]["init_predictions"] = gpytorch_results[
            "init_y_pred"
        ].tolist()
        detailed_result["gpytorch"]["init_std_predictions"] = gpytorch_results[
            "init_y_std"
        ].tolist()
        detailed_result["gpytorch"]["hyperparams"] = gpytorch_results["hyperparams"]

    detailed_result["test_y"] = test_y.tolist()

    # Summary result should only contain core metrics
    summary_result = {
        "function": func_name,
        "dim": dim,
        "n_train": n_train,
        "n_test": n_test,
        "noise_level": noise_level,
        "repeat_id": repeat_id,
        "higp": {
            "init_rmse": higp_init_rmse,
            "rmse": higp_rmse,
            "r2": higp_r2,
            "training_time": higp_results["training_time"],
            "inference_time": higp_results["inference_time"],
            "init_loss": higp_results["init_loss"],
            "final_loss": higp_results["final_loss"],
        },
    }

    if not skip_gpytorch:
        summary_result["gpytorch"] = {
            "init_rmse": gpytorch_init_rmse,
            "rmse": gpytorch_rmse,
            "r2": gpytorch_r2,
            "training_time": gpytorch_results["training_time"],
            "inference_time": gpytorch_results["inference_time"],
            "init_loss": gpytorch_results["init_loss"],
            "final_loss": gpytorch_results["final_loss"],
        }

    return summary_result, detailed_result


def main():
    parser = argparse.ArgumentParser(description="Run Virtual Library benchmark tests")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to JSON configuration file"
    )
    parser.add_argument(
        "--output_dir", type=str, default="results", help="Directory to save results"
    )
    parser.add_argument(
        "--higp-only",
        action="store_true",
        help="Only run HiGP experiments (skip GPyTorch)",
    )
    args = parser.parse_args()

    # Load configuration
    with open(args.config, "r") as f:
        config = json.load(f)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = config.get("name", "experiment")

    # Results storage
    all_results = []
    all_detailed = []

    # Get configurations
    functions = config["functions"]
    dimensions = config["dimensions"]
    n_train_configs = config["n_train_configs"]
    n_test = config["n_test"]
    noise_levels = config["noise_levels"]
    n_repeats = config["n_repeats"]
    seed = config.get("seed", 42)

    higp_params = config.get("higp_params", {})
    gpytorch_params = config.get("gpytorch_params", {})

    # Count total experiments
    total_exps = 0
    for func in functions:
        for dim in dimensions:
            func_info = get_function_info(func)
            if func_info and (
                func_info["supports_dims"] == "any" or dim == func_info["supports_dims"]
            ):
                dim_key = str(dim)
                if dim_key in n_train_configs:
                    for n_train in n_train_configs[dim_key]:
                        for noise in noise_levels:
                            total_exps += n_repeats

    print(f"Running {total_exps} experiments from config: {args.config}")
    print(f"Functions: {functions}")
    print(f"Dimensions: {dimensions}")
    print(f"Noise levels: {noise_levels}")
    print(f"Repeats per config: {n_repeats}")
    if args.higp_only:
        print("Mode: HiGP only (skipping GPyTorch)")
    else:
        print("Mode: Comparing HiGP and GPyTorch")
    print()

    # Run experiments
    exp_count = 0
    for func in functions:
        for dim in dimensions:
            # Check if dimension is valid for this function
            func_info = get_function_info(func)
            if not func_info:
                continue
            if (
                func_info["supports_dims"] != "any"
                and dim != func_info["supports_dims"]
            ):
                continue

            # Get training sizes for this dimension
            dim_key = str(dim)
            if dim_key not in n_train_configs:
                print(f"Warning: No training config for dimension {dim}")
                continue

            for n_train in n_train_configs[dim_key]:
                for noise in noise_levels:
                    for repeat in range(n_repeats):
                        exp_count += 1
                        print(
                            f"Experiment {exp_count}/{total_exps}: {func} dim={dim} n_train={n_train} noise={noise} repeat={repeat+1}"
                        )

                        result, detailed = run_single_experiment(
                            func,
                            dim,
                            n_train,
                            n_test,
                            noise,
                            higp_params,
                            gpytorch_params,
                            seed,
                            repeat,
                            skip_gpytorch=args.higp_only,
                        )

                        if result is not None:
                            all_results.append(result)
                            all_detailed.append(detailed)

                        print()

    # Save results
    summary_path = os.path.join(args.output_dir, f"{exp_name}_summary_{timestamp}.json")
    detailed_path = os.path.join(
        args.output_dir, f"{exp_name}_detailed_{timestamp}.json"
    )

    with open(summary_path, "w") as f:
        json.dump({"config": config, "results": all_results}, f, indent=2)

    with open(detailed_path, "w") as f:
        json.dump({"config": config, "results": all_detailed}, f, indent=2)

    print(f"Results saved to:")
    print(f"  Summary: {summary_path}")
    print(f"  Detailed: {detailed_path}")

    # Print summary statistics
    print("\n=== Summary Statistics ===")
    for func in functions:
        func_results = [r for r in all_results if r["function"] == func]
        if not func_results:
            continue

        print(f"\n{func}:")

        # Average across repeats for each configuration
        configs = {}
        for r in func_results:
            key = (r["dim"], r["n_train"], r["noise_level"])
            if key not in configs:
                configs[key] = []
            configs[key].append(r)

        for (dim, n_train, noise), config_results in sorted(configs.items()):
            higp_init_rmse = sum(r["higp"]["init_rmse"] for r in config_results) / len(
                config_results
            )
            higp_rmse = sum(r["higp"]["rmse"] for r in config_results) / len(
                config_results
            )
            higp_time = sum(r["higp"]["training_time"] for r in config_results) / len(
                config_results
            )

            print(f"  dim={dim}, n_train={n_train}, noise={noise}:")
            print(
                f"    HiGP:     Init RMSE={higp_init_rmse:.4f}, Final RMSE={higp_rmse:.4f}, Time={higp_time:.3f}s"
            )

            # Only print GPyTorch results if they exist
            if "gpytorch" in config_results[0]:
                gpytorch_init_rmse = sum(
                    r["gpytorch"]["init_rmse"] for r in config_results
                ) / len(config_results)
                gpytorch_rmse = sum(
                    r["gpytorch"]["rmse"] for r in config_results
                ) / len(config_results)
                gpytorch_time = sum(
                    r["gpytorch"]["training_time"] for r in config_results
                ) / len(config_results)
                print(
                    f"    GPyTorch: Init RMSE={gpytorch_init_rmse:.4f}, Final RMSE={gpytorch_rmse:.4f}, Time={gpytorch_time:.3f}s"
                )


if __name__ == "__main__":
    main()
