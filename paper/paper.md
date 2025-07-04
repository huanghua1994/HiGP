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

Gaussian Processes (GPs) [@Rasmussen:2005] are flexible, nonparametric Bayesian models widely used for regression and classification tasks due to their ability to capture complex data patterns and provide uncertainty quantification. Traditional GP implementations often face challenges in scalability as data sizes increase. To address these challenges, HiGP, a high-performance Python package, is designed for efficient Gaussian Process regression and classification across datasets of varying sizes. HiGP uses iterative solvers to access large GP computations. It implements matrix-vector (MatVec) and matrix-matrix (MatMul) multiplication algorithms tailored to kernel matrices by exploiting their hierarchical low-rank structure [@Xing:2019; @Huang:2020; @Cai:2023]. To improve the convergence of iterative methods, HiGP also integrates the Adaptive Factorized Nyström (AFN) preconditioner [@Zhao:2024] which is designed to be robust across a wide range of kernel hyperparameter values used during hyperparameter optimization. HiGP can be used with PyTorch and other Python packages, allowing it to be incorporated into common machine learning and data analysis workflows.

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
Using preconditioned iterative methods with preconditioner $\mathbf{M} \approx \widehat{\mathbf{K}}$ is a common option [@Aune:2014, @Zhang:2024, @Hensman:2013, @Wilson:2015, @Pleiss:2018, @Wenger:2022, @Chen:2023]. In this approach, $\widehat{\mathbf{K}}^{-1}\mathbf{y}$ is approximated via the preconditioned conjugate gradient (PCG) method [@Saad:2003]. To handle the logarithmic determinant and trace terms, they are first rewritten as 
$$
\log|\widehat{\mathbf{K}}|=\log|\mathbf{M}| + \log|\mathbf{M}^{-1/2}\widehat{\mathbf{K}}\mathbf{M}^{-1/2}|, \tag{1}
$$
$$
\text{tr}{(\widehat{\mathbf{K}}^{-1} \frac{\partial \widehat{\mathbf{K}}}{\partial \theta})}={\text{tr}}\left({{\mathbf{M}}^{-1}}\frac{\partial {\mathbf{M}}}{\partial \theta}\right)+
 {\text{tr}}\left({\widehat{\mathbf{K}}^{-1}}\frac{\partial \widehat{\mathbf{K}}}{\partial \theta}-{{\mathbf{M}}^{-1}}\frac{\partial {\mathbf{M}}}{\partial \theta}\right). \tag{2}
 $$
 The second component of each new expression is then estimated using the stochastic Lanczos quadrature [@Ubaru:2017] and the Hutchinson estimator [@Hutchinson:1989, Meyer:2021], respectively. The AFN preconditioner is used in HiGP to speed up the optimization process.

# Statement of Need

Multiple GP packages have been released in recent years to utilize modern hardware and handle larger datasets, including GPyTorch [@Gardner:2018], GPflow [@GPflow:2017, @GPflow:2020], GPy [@GPy:2014], and other packages. The motivation for HiGP's development is to utilize new numerical algorithms and parallel computing techniques to reduce computational complexity and to improve the computation efficiency of iterative methods in GP model training. Compared to existing packages, HiGP has three main advantages and contributions.

Firstly, HiGP addresses the efficiency of MatVec, the most performance-critical operation in iterative methods. For large 2D or 3D datasets (e.g. spatial data), the dense kernel matrix is compressed into a $\mathcal{H}^2$ matrix [@Hackbusch:2000, @Hackbusch:2002] in HiGP, resulting in $\mathcal{O}(n)$ storage and computation costs. For large high-dimensional datasets, HiGP computes small blocks of the kernel matrix on-the-fly and immediately uses these blocks in MatVec before discarding them instead of storing them in memory. This on-the-fly mode allows HiGP to handle extremely large datasets on a computer with moderate memory size.

Secondly, HiGP adopts a scalable computational approach: iterative solvers with a robust preconditioner. HiGP adopts the newly proposed AFN preconditioner, which is designed for robust preconditioning of kernel matrices. Numerical experiments demonstrate that AFN can significantly improve the accuracy and robustness of iterative solvers for kernel matrix systems.

Lastly, HiGP uses accurate and efficient hand-coded gradient calculations. GPyTorch relies on the automatic differentiation (autodiff) provided in PyTorch to calculate gradients (Equation (2)). Although autodiff is convenient and flexible, it is very inefficient when used to evaluate derivatives for iterative methods, due to long chain rule expressions resulting from the iterations.

# Design and Implementation

We implemented HiGP in Python 3 and C++ with the goal of providing both a set of ready-to-use out-of-the-box Python interfaces for regular users and a set of reusable high-performance multithreading computational primitives for advanced users. 

The HiGP C++ code implements all performance-critical operations, including populating the matrix block $\mathbf{K}(X, Y; l)$ and $\partial \mathbf{K}(X, Y; l) / \partial l$, computing $\tilde{\mathbf{K}} = f^2 \mathbf{K}(X, Y; l) + s \mathbf{I}$, and matrix products $\tilde{\mathbf{K}} \times B$ and $(\partial \tilde{\mathbf{K}} / \partial \theta) \times B$ in dense or $\mathcal{H}^2$ matrix form, preconditioned conjugate gradient (PCG) solver, and computations of loss and gradients in GP regression and GP classification.

HiGP Python code wraps the C++ units into four basic Python modules: `krnlmatmodule` for computing kernel matrices and its derivatives, `precondmodule` for PCG solver with AFN preconditioner, `gprproblemmodule` and `gpcproblemmodule` for computing the the loss and gradient for GP regression and classification. Advanced users can utilize `krnlmatmodule` and `precondmodule` modules to develop new algorithms for kernel matrices. The two modules `gprproblemmodule` and `gpcproblemmodule` allow a user to train a GP model with any gradient-based optimizer.

We further implemented two high-level modules `GPRModel` and `GPCModel` to simplify the training and use of GP models. These modules register the hyperparameters used in GP regression/classification as PyTorch parameters and set the gradients of PyTorch parameters in each step model the PyTorch optimizer. Listing 1 shows an example of defining and training a GP regression and using the trained model for prediction in just eight lines of code. 

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

We note that the HiGP Python interfaces are *stateless*. This design aims to simplify the interface and decouple different operations. A user can train and use different GP models with the same or different data and configurations in the same file.
