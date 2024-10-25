# HiGP

Gaussian processes (GPs) are a set of flexible, non-parametric Bayesian methods for modeling complex data. HiGP is a high-performance Python package for GP regression (GPR) and GP classification (GPC). HiGP uses multiple new algorithms and techniques to accelerate GP model training and inference for any size of data sets:

* Highly-optimized C++ implementations of computation kernels, including kernel matrix evaluation, Krylov subspace solvers, preconditioner, and gradient calculations.
* The [Adaptive Factorized Nystrom](https://epubs.siam.org/doi/10.1137/23M1565139) preconditioner for efficient and robust gradient calculations.
* $\mathcal{H}^2$ matrix algorithms from the [H2Pack](https://dl.acm.org/doi/abs/10.1145/3412850) library for efficient handling of 2D/3D spatial data.
* On-the-fly running mode for handling large size datasets without using a lot of memory.
* Works with PyTorch optimizer -- you can use HiGP in your PyTorch-based data science workflow easily!

To start using HiGP, please refer to the online documentation:

1. [Basic usage of HiGP](https://github.com/huanghua1994/HiGP/blob/main/docs/1-Basic-usage-of-HiGP.md)
2. [Advanced usage of HiGP](https://github.com/huanghua1994/HiGP/blob/main/docs/2-Advanced-usage-of-HiGP.md)
3. [API reference](https://github.com/huanghua1994/HiGP/blob/main/docs/3-API-reference.md)
4. [Developer information](https://github.com/huanghua1994/HiGP/blob/main/docs/4-Developer-information.md)

HiGP is developed by:

* Hua Huang (huangh1994@outlook.com)
* Tianshi Xu (tianshi.xu@emory.edu)
* Yuanzhe Xi (yuanzhe.xi@emory.edu)
* Edmond Chow (echow@cc.gatech.edu)

Please feel free to contact us if you have any questions. Please contact Hua Huang and Tianshi Xu for questions related to package usage, features, and development.
