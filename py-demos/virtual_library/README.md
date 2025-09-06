# Virtual Library Benchmark Framework

A benchmark framework for comparing HiGP and GPyTorch performance on synthetic test functions.

## Quick Start

```bash
# Small problems (3000 points) comparing accuracy of HiGP and GPyTorch
python test_vl.py --config configs/test00_accuracy.json --output_dir results

# Larger problems with dimension <=3 (50000/150000 points), HiGP only
python test_vl.py --config configs/test01_high_dimension.json --output_dir results --higp-only

# Rastrigin 2D/20D (30000 points), matern32 kernel, comparing efficiency of HiGP and GPyTorch
python test_vl.py --config configs/test02_rastrigin_matern32.json --higp-only --output_dir results

# Rosenbrock 2D/5D/20D (30000 points), rbf kernel, comparing efficiency of HiGP and GPyTorch
python test_vl.py --config configs/test03_rosenbrock_rbf.json --output_dir results
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