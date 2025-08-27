# Virtual Library Benchmark Framework

A comprehensive benchmarking framework for comparing HiGP and GPyTorch performance on synthetic test functions.

## Quick Start

```bash
# Small problems (3000 points) comparing accuracy of HiGP and GPyTorch
# Exact kernel matrices are formed and stored in memory
# Suitable for accuracy comparison rather than speed benchmarking
python test_vl.py --config configs/test00_accuracy.json --output_dir results

# Larger problems with dimension > 3 (30000 points) comparing HiGP and GPyTorch
# HiGP uses on-the-fly kernel matrix computations for memory efficiency
# Expected to show speed advantages for HiGP
python test_vl.py --config configs/test01_high_dimension.json --output_dir results

# Large-scale problems with dimension <= 3 (160,000 points), HiGP-only
# HiGP uses H² hierarchical matrices
python test_vl.py --config configs/test02_low_dimension.json --higp-only --output_dir results

# Run comprehensive benchmark suite
python test_vl.py --config configs/test03_full_benchmark.json --output_dir results
```

## Test Functions

- **Rosenbrock**: Classic optimization benchmark (N-dimensional)
- **Rastrigin**: Many local minima, one global minimum
- **Branin**: 2D only, multiple global minima

## Configuration

JSON configs control:
- `functions`: Which test functions to use
- `dimensions`: Input dimensions to test
- `n_train_configs`: Training set sizes per dimension
- `noise_levels`: Gaussian noise std to add
- `n_repeats`: Number of random repeats
- `higp_params` / `gpytorch_params`: Algorithm-specific settings

## Output

Two files are generated:
- `*_summary.json`: Core metrics (RMSE, R², timing)
- `*_detailed.json`: Full predictions and loss history

## Key Metrics

- **RMSE**: Root Mean Square Error (lower is better)
- **R²**: Coefficient of determination (higher is better)  
- **NLL**: Negative Log-Likelihood (lower is better)
- **Training/Inference Time**: Wall-clock seconds

## Notes

- Initial RMSE shows performance before optimization
- GPyTorch uses BBMM (training) + LOVE (prediction)
- HiGP uses CG + AFN consistently