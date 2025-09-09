# Experimental Evaluation

## Experimental Setup

### Experimental Environment

All experiments were conducted on an Ubuntu 20.04.4 LTS machine equipped with a 24-core 3.0 GHz Intel Xeon Gold 6248R CPU. We used PyTorch 2.8.0 and GPyTorch 1.14 for the experiments.

### Datasets and Data Processing

**UCI Benchmark Datasets:**
- Bike Sharing (15D) 
- 3D Road (2D)

**Virtual Library Functions:**
- Rosenbrock
- Rastrigin
- Branin

All datasets were normalized using Z-score normalization (\(\mu=0\), \(\sigma=1\)) applied to both features and targets using statistics from the training set. The results represent averages over three independent runs to ensure statistical reliability.

### Configuration Parameters

Both HiGP and GPyTorch were configured with identical computational budgets to ensure a fair comparison:

- **CG iterations**: 20 (training), 50 (prediction)
- **Preconditioner/AFN rank**: 10 (training), 100 (prediction)
- **Optimizer**: Adam with a learning rate of 0.01
- **Precision**: float32

### Evaluation Strategy

1. **Small-scale experiments** (n ≤ 3,000): Validate that HiGP achieves equivalent GP accuracy compared to GPyTorch
2. **Large-scale experiments** (n ≥ 30,000): Compare computational performance under fixed computational budget constraints

## Results

### Small-scale Accuracy Validation

Small-scale experiments confirmed that HiGP achieves comparable prediction accuracy to GPyTorch.

| Dataset | n\_train | Kernel | HiGP Mode | HiGP Final RMSE | GPyTorch Final RMSE |
|---------|---------|--------|-----------|-----------------|---------------------|
| Bike | 3,000 | RBF | dense | 0.0290 | 0.0279 |
| Rosenbrock (5D) | 3,000 | Matern32 | dense | 0.0609 | 0.0669 |

### Large-scale Performance Evaluation

Large-scale experiments focused on computational efficiency under fixed computational budget constraints.

| Dataset | n\_train | Kernel | HiGP Mode | HiGP Time (s) | GPyTorch Time (s) |
|---------|---------|--------|-----------|---------------|------------------|
| Road3D | 50,000 | RBF | H2 | 420.7 | 34,278 |
| Road3D | 150,000 | RBF | H2 | 927.3 | — |
| Branin | 50,000 | RBF | H2 | 296.3 | - |
| Branin | 150,000 | RBF | H2 | 602.0 | — |
| Rastrigin (2D) | 30,000 | Matern32 | H2 | 213.2 | 3,648 |
| Rastrigin (20D) | 30,000 | Matern32 | on-the-fly | 527.2 | 3,376 |
| Rosenbrock (2D) | 30,000 | RBF | H2 | 176.6 | 3,831 |
| Rosenbrock (20D) | 30,000 | RBF | on-the-fly | 520.4 | 3,633 |

### Summary

1. Small-scale experiments demonstrate that HiGP achieves comparable prediction accuracy to GPyTorch across diverse problem types.

2. Large-scale experiments demonstrate that HiGP achieves significant computational advantages.

3. HiGP's H2 matrix mode enables the efficient processing of datasets with large sizes.
