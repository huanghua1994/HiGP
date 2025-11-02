# 5. Performance tests

We conducted performance tests on an Ubuntu 20.04 LTS machine with a 24-core 3.0 GHz Intel Xeon Gold 6248R CPU. We used PyTorch 2.8.0, GPyTorch 1.14, and HiGP version 2025.8.21 (git commit 8942631cd9fb4f213afd25032247e689da2ee2c0) for the tests.

We tested two data sets from the [UCI Machine Learning Datasets](https://archive.ics.uci.edu/datasets): the "Bike Sharing" and the "3D Road Network" data sets. We also tested three synthetic target functions from the [Virtual Library of Simulation Experiments](https://www.sfu.ca/~ssurjano) with randomly sampled data points: Rosenbrock, Rastrigin, and Branin. All datasets were normalized with Z-score normalization ($\mu=0$, $\sigma=1$) applied to both features and targets using statistics from the training set. The results represent averages over three independent runs for statistical reliability. Please check the code in [py-demos/uci_benchmark](https://github.com/huanghua1994/HiGP/tree/main/py-demos/uci_benchmark) and [py-demos/virtual_library](https://github.com/huanghua1994/HiGP/tree/main/py-demos/virtual_library) for more details of the tests.

Both HiGP and GPyTorch were configured with identical computational budgets to ensure a fair comparison:

- CG iterations: 20 (training), 50 (prediction)
- Preconditioner/AFN rank: 10 (training), 100 (prediction)
- Optimizer: Adam with a learning rate of 0.01
- Precision: 32-bit floating point (FP32)

We used two evaluation strategies:

1. Small-scale experiments for n ≤ 3,000: to validate that HiGP achieves equivalent GP accuracy compared to GPyTorch
2. Large-scale experiments for n ≥ 30,000: to compare computational performance under fixed computational budget constraints

The following table lists the results of small-scale experiments. These results confirm that HiGP achieves comparable prediction accuracy to GPyTorch.

| Dataset | n\_train | Kernel | HiGP Mode | HiGP Final RMSE | GPyTorch Final RMSE |
|:-------:|--------:|:------:|:---------:|--------------:|-----------------:|
| Bike | 3,000 | RBF | dense | 0.0290 | 0.0279 |
| Rosenbrock (5D) | 3,000 | Matern32 | dense | 0.0609 | 0.0669 |

The following table lists the results of large-scale experiments. These results show that HiGP achieves significant computational advantages over GPyTorch under the same computational budget constraints, and HiGP's H2 matrix mode enables the efficient processing of datasets with large sizes.

| Dataset | n\_train | Kernel | HiGP Mode | HiGP Time (s) | GPyTorch Time (s) |
|:-------:|--------:|:------:|:---------:|--------------:|-----------------:|
| Road3D | 50,000 | RBF | H2 | 420.7 | 34,278 |
| Road3D | 150,000 | RBF | H2 | 927.3 | — |
| Branin | 50,000 | RBF | H2 | 296.3 | - |
| Branin | 150,000 | RBF | H2 | 602.0 | — |
| Rastrigin (2D) | 30,000 | Matern32 | H2 | 213.2 | 3,648 |
| Rastrigin (20D) | 30,000 | Matern32 | on-the-fly | 527.2 | 3,376 |
| Rosenbrock (2D) | 30,000 | RBF | H2 | 176.6 | 3,831 |
| Rosenbrock (20D) | 30,000 | RBF | on-the-fly | 520.4 | 3,633 |
