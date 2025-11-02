---
title: 'HiGP: A high-performance Python package for Gaussian Processes'
tags:
  - Python
  - Gaussian process
  - iterative methods
  - machine learning
authors:
  - name: Hua Huang
    orcid: 0000-0003-1060-5639
    equal-contrib: true
    affiliation: 1
  - name: Tianshi Xu
    orcid: 0000-0003-3119-1957
    equal-contrib: true
    affiliation: 2
  - name: Yuanzhe Xi
    orcid: 0000-0002-4720-9915
    affiliation: 2
  - name: Edmond Chow
    orcid: 0000-0003-0474-3752
    corresponding: true
    affiliation: 1
affiliations:
 - name: School of Computational Science, Engineering, Georgia Institute of Technology, USA
   index: 1
 - name: Department of Mathematics, Emory University, USA
   index: 2
date: 3 July 2025
bibliography: paper.bib
tags:
  - Gaussian Process
  - iterative method
  - preconditioner
---

# Summary

Gaussian Processes (GPs) [@Rasmussen:2005] are flexible, nonparametric Bayesian models widely used for regression and classification because of their ability to capture complex data patterns and quantify predictive uncertainty. However, the $\mathcal{O}(n^3)$ computational cost of kernel matrix operations poses a major obstacle to applying GPs at scale. HiGP is a high-performance Python package designed to overcome these scalability limitations through advanced numerical linear algebra and hierarchical kernel representations.  It integrates $\mathcal{H}^2$ matrices to achieve near-linear complexity in both storage and computation for spatial datasets, supports on-the-fly kernel evaluation to avoid explicit storage in large-scale problems, and incorporates a robust Adaptive Factorized Nyström (AFN) preconditioner [@Zhao:2024] that accelerates convergence of iterative solvers across a broad range of kernel spectra. These computational kernels are implemented in C++ for maximum performance and exposed through Python interfaces, enabling seamless integration with modern machine learning workflows. HiGP also includes analytically derived gradient computations for efficient hyperparameter optimization, avoiding the inefficiencies of automatic differentiation in iterative solvers. By serving as a reusable numerical engine, HiGP complements existing GP frameworks such as GPJax [@Pinder:2022], KeOps [@Charlier:2021], and GaussianProcesses.jl [@Fairbrother:2022], providing a reliable and scalable computational backbone for large-scale Gaussian Process regression and classification.

# Gaussian Processes

For training points $\mathbf{X} \in \mathbb{R}^{n \times d}$, a noisy training observation set $\mathbf{y} \in \mathbb{R}^{n}$, and testing points $\mathbf{X}_\ast \in \mathbb{R}^{m \times d}$, a standard GP model assumes that the noise-free testing observations $\mathbf{y}_\ast \in \mathbb{R}^{m}$ follow a joint Gaussian distribution that depends on a set of parameters, including scale $f$, noise level $s$, and kernel parameters $l$. The GP model finds the optimal parameters $\Theta:=(s,f,l)$ by minimizing the negative log marginal likelihood:
$$
L(\Theta) = \frac{1}{2} \left( \mathbf{y}^{\top} \widehat{\mathbf{K}}^{-1} \mathbf{y} 
+ \log|\widehat{\mathbf{K}}| + n\log 2\pi \right),
$$
where $\widehat{\mathbf{K}}$ denotes the regularized kernel matrix. An optimization process usually requires the gradient of $L(\Theta)$:
$$
\frac{\partial L}{\partial \theta} =
\frac{1}{2} \left(-\mathbf{y}^{\top} \widehat{\mathbf{K}}^{-1} 
\frac{\partial \widehat{\mathbf{K}}}{\partial \theta} \widehat{\mathbf{K}}^{-1}\mathbf{y} +
\text{tr}{\left( \widehat{\mathbf{K}}^{-1} \frac{\partial \widehat{\mathbf{K}}}{\partial \theta} \right)}\right),
\quad \theta \in \Theta.
$$
Using preconditioned iterative methods with preconditioner $\mathbf{M} \approx \widehat{\mathbf{K}}$ is a common option [@Aune:2014; @Zhang:2024; @Hensman:2013; @Wilson:2015; @Pleiss:2018; @Wenger:2022; @Chen:2023]. In this approach, $\widehat{\mathbf{K}}^{-1}\mathbf{y}$ is approximated via the preconditioned conjugate gradient (PCG) method [@Saad:2003]. To handle the logarithmic determinant and trace terms, they are first rewritten as
$$
\log|\widehat{\mathbf{K}}|=\log|\mathbf{M}| + \log|\mathbf{M}^{-1/2}\widehat{\mathbf{K}}\mathbf{M}^{-1/2}|, \tag{1}
$$

\begin{equation}
\label{eq:trace}
\text{tr}{(\widehat{\mathbf{K}}^{-1} \frac{\partial \widehat{\mathbf{K}}}{\partial \theta})}={\text{tr}}\left({{\mathbf{M}}^{-1}}\frac{\partial {\mathbf{M}}}{\partial \theta}\right)+
 {\text{tr}}\left({\widehat{\mathbf{K}}^{-1}}\frac{\partial \widehat{\mathbf{K}}}{\partial \theta}-{{\mathbf{M}}^{-1}}\frac{\partial {\mathbf{M}}}{\partial \theta}\right). \tag{2}
\end{equation}
The second component of each new expression is then estimated using the stochastic Lanczos quadrature [@Ubaru:2017] and the Hutchinson estimator [@Hutchinson:1989; Meyer:2021], respectively.

# Statement of Need

The Gaussian Process (GP) community has advanced rapidly in recent years, developing scalable inference frameworks and more efficient kernel representations. Modern libraries such as GPyTorch [@Gardner:2018], GPflow [@GPflow:2017; @GPflow:2020], GPJax [@Pinder:2022], KeOps [@Charlier:2021], and GaussianProcesses.jl [@Fairbrother:2022] leverage GPUs and automatic differentiation to perform GP inference efficiently on moderately large datasets. Concurrently, new algorithms, including preconditioned optimization methods [@Wenger:2022], alternating-projection solvers [@Wu:2024], GPU-accelerated Vecchia approximations for spatial data [@Pan:2024], robust relevance-pursuit inference [@Ament:2024], and latent Kronecker formulations for structured covariance matrices [@Lin:2025], have further improved the scalability and robustness of GP models. Yet, most existing frameworks emphasize modeling flexibility and seamless integration with autodiff ecosystems, rather than optimizing the low-level numerical routines that dominate runtime for very large or ill-conditioned kernel systems. HiGP is designed to address this computational gap by focusing on the numerical core of GP inference. It provides robust, scalable, and hardware-efficient implementations of kernel algebra, preconditioned iterative solvers, and gradient computations, offering three primary contributions.

Firstly, HiGP addresses the efficiency of MatVec, the most performance-critical operation in iterative methods. For large 2D or 3D datasets, the dense kernel matrix is compressed into a $\mathcal{H}^2$ matrix [@Hackbusch:2000; @Hackbusch:2002] in HiGP, resulting in $\mathcal{O}(n)$ storage and computation costs. For large high-dimensional datasets, HiGP computes small kernel matrix blocks on-the-fly and immediately uses them in MatVec and discards them, which allows HiGP to handle extremely large datasets with a moderate memory size.

Secondly, HiGP uses iterative solvers with the newly proposed AFN preconditioner [@Zhao:2024], which is designed for robust preconditioning of kernel matrices. Experiments demonstrate that AFN can significantly improve the accuracy and robustness of iterative solvers for kernel matrix systems. Furthermore, AFN and $\mathcal{H}^2$ matrix computation rely on evaluating many small kernel matrices in parallel, which is easily handled in C++ but would incur large overhead in Python, making implementation in other libraries such as GPyTorch or GPFlow more challenging.

Lastly, HiGP uses accurate and efficient hand-coded gradient calculations. GPyTorch relies on the automatic differentiation (autodiff) provided in PyTorch to calculate gradients (\autoref{eq:trace}). However, autodiff can be inefficient and inaccurate for computing the gradient of the preconditioner, so we use hand-coded gradient calculations for better performance and accuracy.

The HiGP documentation[^1] provides a comparison of the accuracy and performance of HiGP version 2025.8.21 (git commit 8942631) and GPyTorch version 1.14. The tests were performed on one node (shared memory) with a 24-core 3.0 GHz Intel Xeon Gold 6248R CPU, using the "Bike Sharing" and "3D Road Network" data sets from the UCI Machine Learning Datasets and three synthetic target functions "Rosenbrock", "Rastrigin", and "Branin" from the Virtual Library of Simulation Experiments; HiGP is 539\% to 8061\% faster than GPyTorch.

[^1]: https://github.com/huanghua1994/HiGP/blob/main/docs/5-Performance-tests.md

# Design and Implementation

We implemented HiGP in Python 3 and C++ with the goal of providing both a set of ready-to-use out-of-the-box Python interfaces for regular users and a set of reusable high-performance shared-memory multithreading computational primitives for advanced users. The HiGP C++ code implements all performance-critical operations. The HiGP Python code wraps the C++ units into four basic Python modules: `krnlmatmodule` for computing kernel matrices and its derivatives, `precondmodule` for PCG solver with AFN preconditioner, `gprproblemmodule` and `gpcproblemmodule` for computing the the loss and gradient for GP regression and classification. The two modules `gprproblemmodule` and `gpcproblemmodule` allow a user to train a GP model with any gradient-based optimizer.

We further implemented two high-level modules `GPRModel` and `GPCModel` using PyTorch parameter registration and optimizer to simplify the training and use of GP models. Listing 1 shows an example of defining and training a GP regression and using the trained model for prediction.

```python
# Listing 1: HiGP example code of training and using a GPR model
gprproblem = higp.gprproblem.setup(data=train_x, label=train_y, 
                                   kernel_type=higp.GaussianKernel)
model = higp.GPRModel(gprproblem)
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
for i in ranges(max_steps):
    loss = model.calc_loss_grad()
    optimizer.step()
params = model.get_params()
pred = higp.gpr_prediction(train_x, train_y, test_x, 
                           higp.GaussianKernel, params)
```

We note that the HiGP Python interfaces (except for `GPRModel` and `GPCModel` models) are *stateless*. This design aims to simplify the interface and decouple different operations. A user can train and use different GP models with the same or different data and configurations in the same file.

# References
