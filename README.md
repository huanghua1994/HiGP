# HiGP

**HiGP** is a high-performance Python package for using Gaussian processes (GPs) with large datasets.
Its functionality includes estimating GP hyperparameters, GP regression, and GP classification.
Under the hood, it uses highly-optimized, multithreaded C++ code, implementing iterative solvers
and a novel *preconditioner* designed for large datasets.
It works with PyTorch optimizers, and you can easily use HiGP in your PyTorch-based data science workflow.

HiGP includes:

* Vectorized kernel matrix computations.
* On-the-fly running mode for handling large size datasets without using a lot of memory.
* Krylov subspace iterative solvers, such as conjugate gradients.
* The [Adaptive Factorized Nystrom](https://epubs.siam.org/doi/10.1137/23M1565139) preconditioner
  to accelerate the iterative solvers.
* Linear scaling hierarchical matrix algorithms from [H2Pack](https://dl.acm.org/doi/abs/10.1145/3412850)
  for handling large scale 2D/3D spatial data.
* Efficient gradient calculations for the negative log marginal likelihood, using preconditioned iterative solvers and preconditioned stochastic trace estimation.
* Acceleration with GPUs is coming soon!

To start using HiGP, refer to the online documentation:

1. [Basic usage of HiGP](https://github.com/huanghua1994/HiGP/blob/main/docs/1-Basic-usage-of-HiGP.md)
2. [Advanced usage of HiGP](https://github.com/huanghua1994/HiGP/blob/main/docs/2-Advanced-usage-of-HiGP.md)
3. [API reference](https://github.com/huanghua1994/HiGP/blob/main/docs/3-API-reference.md)
4. [Developer information](https://github.com/huanghua1994/HiGP/blob/main/docs/4-Developer-information.md)

HiGP is developed by:

* Hua Huang (huangh1994@outlook.com)
* Tianshi Xu (tianshi.xu@emory.edu)
* Yuanzhe Xi (yuanzhe.xi@emory.edu)
* Edmond Chow (echow@cc.gatech.edu)

We welcome your questions.
Please contact Hua Huang and Tianshi Xu specifically for questions related to package usage, features, and development.

If you use HiGP, please cite the following paper:

```text
@misc{HiGP2025,
    author = {Hua Huang and Tianshi Xu and Yuanzhe Xi and Edmond Chow},
    title = {{HiGP}: A high-performance {Python} package for {Gaussian} Process},
    year = {2025},
    eprint = {arXiv:2503.02259},
}
```
