```text
% This text block is just to allow some editors to render the markdown file correctly. Remove the text block before submission. 
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
```

# Summary

Gaussian Processes (GPs) [@Rasmussen:2005] are flexible, nonparametric Bayesian models widely used for regression and classification tasks due to their ability to capture complex data patterns and provide uncertainty quantification. Traditional GP implementations often face challenges in scalability as data sizes increase. To address these challenges, HiGP, a high-performance Python package, is designed for efficient Gaussian Process regression and classification across datasets of varying sizes. HiGP uses  iterative solvers to access large GP computations. It implements matrix-vector (MatVec) and matrix-matrix (MatMul) multiplication algorithms tailored to kernel matrices by exploiting their hierarchical low-rank structure [@Xing:2019; @Huang:2020; @Cai:2023]. To improve the convergence of iterative methods, HiGP also integrates the Adaptive Factorized Nyström (AFN) preconditioner [@Zhao:2024] which is designed to be robust across a wide range of kernel hyperparameter values used during hyperparameter optimization. HiGP can be used with PyTorch and other Python packages, allowing it to be incorporated into common machine learning and data analysis workflows.

# Gaussian Processes

For training points $\mathbf{X} \in \mathbb{R}^{n \times d}$, a noisy training observation set $\mathbf{y} \in \mathbb{R}^{n}$, and testing points $\mathbf{X}_\ast \in \mathbb{R}^{m \times d}$, a standard GP model assumes that the noise-free testing observations $\mathbf{y}_\ast \in \mathbb{R}^{m}$ follow the joint distribution:
$$
\begin{bmatrix}
\mathbf{y} \\
\mathbf{y}_\ast
\end{bmatrix}
\sim
\mathcal{N} 
\begin{pmatrix}
\mathbf{0},
f^2
\begin{bmatrix}
    \kappa(\mathbf{X}, \mathbf{X}) + s \mathbf{I} & \kappa(\mathbf{X}, \mathbf{X}_{\ast}) \\ 
    \kappa(\mathbf{X}_{\ast}, \mathbf{X}) & \kappa(\mathbf{X}_{\ast}, \mathbf{X}_{\ast})
\end{bmatrix}
\end{pmatrix}. \tag{1}
$$
Here, $f > 0$ and $s \ge 0$ are real numbers, $\mathbf{I}$ is the identity matrix, $\kappa(\mathbf{u}, \mathbf{v}): \mathbb{R}^d \times \mathbb{R}^d \rightarrow \mathbb{R}$ is a kernel function, and $\kappa(\mathbf{X}, \mathbf{Y})$ is a kernel matrix with the $(i,j)$-th entry defined as $\kappa(\mathbf{X}_{i,:}, \mathbf{Y}_{j,:})$, where $\mathbf{X}_{i,:}$ denotes the $i$-th row of $\mathbf{X}$. Commonly used kernel functions include the Gaussian kernel (also known as the Radial Basis Function kernel) and Mat\'ern kernels. These kernel functions typically depend on one or more kernel parameters. For example, the Gaussian kernel $\kappa(\mathbf{u}, \mathbf{v}) = \exp(-\|\mathbf{u} - \mathbf{v}\|^2 / (2l^2))$ depends on the parameter $l > 0$, typically known as the length scale.

To find the $s$, $f$, and kernel parameter $l$ that best fit the data, an optimization process is generally required to minimize the negative log marginal likelihood:
$$
L(\Theta) = \frac{1}{2} \left( \mathbf{y}^{\top} \widehat{\mathbf{K}}^{-1} \mathbf{y} 
+ \log|\widehat{\mathbf{K}}| + n\log 2\pi \right), \tag{2}
$$
where $\widehat{\mathbf{K}}$ denotes the regularized kernel matrix $f^2 \kappa(\mathbf{X}, \mathbf{X}) + s \mathbf{I}$ and $\Theta$ denotes the hyperparameter set $(s, f, l)$. An optimization process usually requires the gradient of $L(\Theta)$:
$$
\frac{\partial L}{\partial \theta} =
\frac{1}{2} \left(-\mathbf{y}^{\top} \widehat{\mathbf{K}}^{-1}
\frac{\partial \widehat{\mathbf{K}}}{\partial \theta} \widehat{\mathbf{K}}^{-1}\mathbf{y} +
\text{tr}{\left( \widehat{\mathbf{K}}^{-1} \frac{\partial \widehat{\mathbf{K}}}{\partial \theta} \right)}\right),
\quad \theta \in \Theta. \tag{3}
$$

For small or moderate size datasets, $\widehat{\mathbf{K}}$, $\widehat{\mathbf{K}}^{-1}$, and $\partial \widehat{\mathbf{K}} / \partial \theta$ can be formed explicitly, and Formulas (2) and (3) can be calculated exactly. For large datasets, it is often intractable to populate and store $\widehat{\mathbf{K}}$, $\widehat{\mathbf{K}}^{-1}$, or $\partial \widehat{\mathbf{K}} / \partial \theta$, as these matrices require $\mathcal{O}(n^2)$ space for storage and $\widehat{\mathbf{K}}^{-1}$ requires $\mathcal{O}(n^3)$ arithmetic operations to compute. Instead, using preconditioned iterative methods with preconditioner $M \approx \tilde{K}$ is often a better option [@Aune:2014], [@Zhang:2024], [@Michalis:2009], [@Hensman:2013], [@Wilson:2015], [@Pleiss:2018], [@Wenger:2022], [@Chen:2023]. In this approach, $\mathbf{K}^{-1}\mathbf{y}$ is approximated via the Preconditioned Conjugate Gradient (PCG) method [@Saad:2003]. The trace term $\text{tr}{(\widehat{\mathbf{K}}^{-1} \frac{\partial \widehat{\mathbf{K}}}{\partial \theta})}$ can be rewritten as 
$$
{\text{tr}}\left({{\mathbf{M}}^{-1}}\frac{\partial {\mathbf{M}}}{\partial \theta}\right) + {\text{tr}}\left({\widehat{\mathbf{K}}^{-1}}\frac{\partial \widehat{\mathbf{K}}}{\partial \theta}-{{\mathbf{M}}^{-1}}\frac{\partial {\mathbf{M}}}{\partial \theta}\right),
$$
and the second term can then be estimated by the Hutchinson estimator [@Hutchinson:1989; @Meyer:2021]:
$$
{\text{tr}}
\left(
  {\widehat{\mathbf{K}}^{-1}}\frac{\partial \widehat{\mathbf{K}}}{\partial \theta} - 
  {{\mathbf{M}}^{-1}}\frac{\partial {\mathbf{M}}}{\partial \theta}
\right)
\approx 
\frac{1}{k} \sum_{i=1}^{k} \mathbf{z}_{i}^{\top} 
\left(
  {\widehat{\mathbf{K}}^{-1}}\frac{\partial \widehat{\mathbf{K}}}{\partial \theta} -
  {{\mathbf{M}}^{-1}}\frac{\partial {\mathbf{M}}}{\partial \theta}
\right) \mathbf{z}_{i}, \tag{4}
$$
where $\mathbf{z}_{i} \sim \mathcal{N}(0, I), i = 1, \cdots, k$ are independent random vectors. To estimate $\log|\hat{\mathbf{K}}|$ in Equation (2), we use stochastic Lanczos quadrature [@Ubaru:2017] for $\text{tr}(\log|\hat{\mathbf{K}}|)$. This method needs to sample $k_z$ independent vectors $\mathbf{z}_i \sim \mathcal{N}(0,I), i = 1, \cdots, k_z$ and solve linear systems
$$
\widehat{\mathbf{K}}\mathbf{u}_i = \mathbf{z}_i,\quad i=1,2,\ldots,k_z. \tag{5}
$$
using PCG with preconditioner $M$. Let $T_{z_i}$ denote the tridiagonal matrix in the Lanczos method on matrix $\mathbf{M}^{-1/2}\widehat{\mathbf{K}}\mathbf{M}^{-1/2}$. Using the identity $\log|\mathbf{K}|= \log|\mathbf{M}| + \log|\mathbf{M}^{-1/2}\widehat{\mathbf{K}}\mathbf{M}^{-1/2}|$, we only apply the Hutchinson estimator to approximate the second term:
$$
\log|\mathbf{M}^{-1/2}\widehat{\mathbf{K}}\mathbf{M}^{-1/2}| = \text{tr} \left( \log \mathbf{M}^{-1/2}\widehat{\mathbf{K}}\mathbf{M}^{-1/2} \right) 
\approx \frac{1}{k_z} \sum_{i=1}^{k_z} \|\mathbf{z}_i\|^2 \mathbf{e}_{1}^\top \log(\mathbf{T}_{\mathbf{z}_i}) \mathbf{e}_{1}, \tag{6}
$$
where \mathbf{e}_1 = [1, 0, 0, ..., 0]^\top$. This preconditioned trace estimation approach is particularly efficient when $\log|\mathbf{M}|$ and ${\text{tr}}\left( \mathbf{M}^{-1} \frac{\partial \mathbf{M}}{\partial \theta} \right)$ can be computed efficiently, leveraging the simpler structure of $\mathbf{M}$ compared to that of $\widehat{\mathbf{K}}$. The AFN preconditioner used in HiGP has a triangular factorization structure, which makes such computation efficient.

# Statement of Need

GP research has undergone significant innovations in recent years, including advances in deep Gaussian processes (DGPs) [@Balcan:2016], [@Damianou:2013], [@Jakkala:2021], [@Dutordoir:2021], preconditioned GPs [@Wenger:2022] [@Wagner:2025], and unbiased GPs [@Potapczynski:2021] [@Burt:2021]. Additionally, there has been a growing focus on improving the accuracy and stability of GP models for large datasets as well as accelerating computations in GP using modern hardware like graphics processing units (GPUs). Multiple GP packages have been released in recent years to address these computational challenges. The GPyTorch package [@Gardner:2018] is built on top of PyTorch to leverage GPU computing capabilities. Similarly, GPflow [@GPflow:2017; @GPflow:2020] leverages another deep learning framework, TensorFlow [@TensorFlow:2015], for GPU acceleration. GPy [@GPy:2014] is supported by NumPy [@Harris:2020] with some GPU support.

The motivation for HiGP's development is to utilize new numerical algorithms and parallel computing techniques to reduce computational complexity and to improve computation efficiency of the iterative methods in GP model training. Compared to existing packages, HiGP has three main advantages and contributions.

Firstly, HiGP addresses the efficiency of MatVec, the most performance-critical operation in iterative methods. Traditional methods populate and store $\mathbf{K}$ and $\partial \widehat{\mathbf{K}} / \partial \theta$ for MatVec, but the $\mathcal{O}(n^2)$ storage and computation costs become prohibitive for large datasets, such as when $n \ge 10,000$. HiGP utilizes two methods to address this issue: the $\mathcal{H}^2$ matrix [@Hackbusch:2000, @Hackbusch:2002] and on-the-fly computation mode. For large 2D or 3D datasets (e.g. spatial data), the dense kernel matrix is compressed into a $\mathcal{H}^2$ matrix in HiGP, resulting in $\mathcal{O}(n)$ storage and computation costs. This approach is also used in the smashGP package [@Erlandson:2025]. For large high-dimensional datasets, HiGP computes small blocks of the kernel matrix on-the-fly and immediately uses these blocks in MatVec before discarding them instead of storing them in memory. This on-the-fly mode allows HiGP to handle extremely large datasets on a computer with moderate memory size.

Secondly, HiGP adopts a scalable computational approach: iterative solvers with a robust preconditioner. In GP model training, changes in hyperparameters result in variations in the kernel matrix's spectrum. Direct methods are robust against changes in the matrix spectrum, but the $\mathcal{O}(n^3)$ computational costs make them unaffordable for large datasets. Iterative solvers are sensitive to the matrix spectrum and might fail to provide solutions with the desired accuracy. Existing GP packages usually use simple preconditioners, such as a low-rank Nyström approximate factorization of the kernel matrix [@Gardner:2018]. However, these simple preconditioners may fail when the kernel matix is not low rank, which is the case for certain kernel length scales. HiGP adopts the newly proposed AFN preconditioner, which is designed for robust preconditioning of kernel matrices. Numerical experiments demonstrate that AFN can significantly improve the accuracy and robustness of iterative solvers for kernel matrix systems.

Lastly, HiGP uses accurate and efficient hand-coded gradient calculations. GPyTorch relies on automatic differentiation (autodiff) provided in PyTorch to calculate gradients (Equation~(\ref{eq:loss_grad})). Although autodiff is convenient and flexible, it is very inefficient when used to evaluate derivatives of quantitives that are computed iteratively, e.g., with PCG, due to long chain rule expressions resulting from the iterations.

# Design and Implementation

We implemented HiGP in Python 3 and C++ with the goal of providing both a set of ready-to-use out-of-the-box Python interfaces for regular users and a set of reusable high-performance computational primitives for experienced users. The HiGP C++ part implements five functional units for performance-critical calculations:

1. The `kernel` unit. This unit populates the matrix block $\mathbf{K}(X, Y; l)$ and optionally $\partial \mathbf{K}(X, Y; l) / \partial l$ for two sets of points $X$, $Y$ (typically subsets of the training or testing data), and a length scale $l$.
2. The `dense_kmat` unit. This unit computes the regularized kernel matrix block $\tilde{\mathbf{K}} = f^2 \mathbf{K}(X, Y; l) + s \mathbf{I}$, and matrix products $\tilde{\mathbf{K}} \times B$ and $(\partial \tilde{\mathbf{K}} / \partial \theta) \times B$, where $B$ is a general dense matrix and $\theta \in \{l, f, s\}$ is a hyperparameter.
3. The `h2mat` unit. This unit is similar to the \texttt{dense\_kmat} unit, but only computes the $\mathcal{H}^2$ matrix-matrix multiplication for $\widehat{\mathbf{K}} \times B$ and $(\partial \widehat{\mathbf{K}} / \partial \theta) \times B$, where $\widehat{\mathbf{K}} = f^2 \mathbf{K}(X, X; l) + s \textbf{I}$ is a symmetric regularized kernel matrix.
4. The `solver` unit. This unit implements a PCG method for solving multiple right-hand-side systems simultaneously, employing the AFN preconditioner.
5. The `gp` unit. This unit implements the trace estimator and the computation of the loss and gradients in GP regression and GP classification computations. The loss and gradients can be computed in an "exact" manner using dense matrix factorization, or in a fast and approximate manner using preconditioned iterative solvers and stochastic Lanczos quadrature for trace estimation.

The aforementioned C++ units can be compiled as a standalone library with C language interfaces for secondary development in many programming languages, including C, C++, Python, Julia, and other languages. 

HiGP wraps the C++ units into four basic Python modules:

1. `higp.krnlmatmodule` wraps and calls the C++ `dense_kmat` and `h2mat` units.
2. `higp.precondmodule` wraps and calls the PCG solver with the AFN precondioner; both are in the C++ `solver` unit.
3. `higp.gprproblemmodule` computes the loss and gradient for GP regression.
4. `higp.gpcproblemmodule` computes the loss and gradient for GP classification.

These basic Python modules provide fast access to the high-performance C++ units. Advanced  users can utilize `higp.krnlmatmodule` and `higp.precondmodule` modules to develop new algorithms for kernel matrices. The Python interface allows faster and easier debugging and testing when prototyping new algorithms. The two modules `higp.gprproblemmodule` and `higp.gpcproblemmodule` allow a user to train a GP model with any gradient-based optimizer, allowing HiGP to be adopted in different data science and machine learning workflows.

To further simplify the training and use of GP models, we further implemented two high-level modules, `higp.GPRModel` and `higp.GPCModel`. These modules register the hyperparameters used in GP regression/classification as PyTorch parameters and set the gradients of PyTorch parameters in each step model the PyTorch optimizer. The Listing 1 below shows an example of defining and training a GP regression and using the trained model for prediction in just eight lines of code, where `pred.prediction_mean` and `pred.prediction_stddev` are the predicted mean values and the standard deviation of the predictive distribution for each data point in `test_x.

```python
# Listing 1: HiGP example code of training and using a GPR model
gprproblem = higp.gprproblem.setup(data=train_x, label=train_y, kernel_type=higp.GaussianKernel)
model = higp.GPRModel(gprproblem)
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
for i in ranges(max_steps):
    loss = model.calc_loss_grad()
    optimizer.step()
params = model.get_params()
pred = higp.gpr_prediction(train_x, train_y, test_x, higp.GaussianKernel, params)
```

We note that the HiGP Python interfaces are *stateless*. For example, the same arguments `train_x`, `train_y`, and `higp.GaussianKernel` are passed into two functions `higp.gprproblem.setup` and `higp.gpr_prediction` in Listing 1. This design aims to simplify the interface and decouple different operations. A user can train and use different GP models with the same or different data and configurations in the same file.

# Acknowledgements

TODO

# References
