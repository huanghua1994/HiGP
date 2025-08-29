# UCI Benchmark Framework

A comprehensive benchmarking framework for comparing HiGP and GPyTorch performance on real UCI datasets.

## Quick Start

```bash
# Small Bike dataset test (3000 samples)
python test_uci.py --config configs/test_bike_small.json --output_dir results

# Small Road3D with H² matrix (30000 samples, HiGP only)
python test_uci.py --config configs/test_road3d_small.json --output_dir results --higp-only

# Run comprehensive benchmark on all datasets
python test_uci.py --config configs/test_all_datasets.json --output_dir results
```

## Supported Datasets

### 1. Bike Sharing Dataset
- **Features**: 15
- **Samples**: ~17,000 hourly records

### 2. 3D Road Network Dataset
- **Features**: 2
- **Samples**: ~400,000 points

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