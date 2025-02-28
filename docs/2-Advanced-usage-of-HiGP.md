# 2. Advanced usage of HiGP

## 2.1 Mathematical definitions

Gaussian processes (GPs) are a flexible, non-parametric Bayesian method for modeling complex data and are important in many scientific and engineering fields.

For training data ${\mathbf{X}} \in {\mathbb{R}}^{n \times d}$, noisy training observations ${\mathbf{y}} \in {\mathbb{R}}^{n}$, and testing data $`{\mathbf{X}}_{*} \in {\mathbb{R}}^{m \times d}`$, a standard GP model assumes that the noise-free test target values $`y_* \in R^{m}`$ follow the joint distribution:

```math
\begin{bmatrix}
    {\mathbf{y}} \\
    {\mathbf{y}}_{*} 
\end{bmatrix}
\sim
{\mathcal{N}} 
\begin{pmatrix}
    \mathbf{0},
    f^2
    \begin{bmatrix}
        \kappa({\mathbf{X}},{\mathbf{X}}) + s \mathbf{I} & \kappa({\mathbf{X}},{\mathbf{X}}_{*} )\\ \kappa({\mathbf{X}}_{*} ,{\mathbf{X}}) & \kappa({\mathbf{X}}_{*} ,{\mathbf{X}}_{*} )
    \end{bmatrix}
\end{pmatrix}.
```

Here, $f$ and $s$ are real numbers, and $\kappa({\mathbf{x}},{\mathbf{y}}): {\mathbb{R}}^d \times {\mathbb{R}}^d \rightarrow {\mathbb{R}}$ is a kernel function. Commonly used kernel functions are listed in [Section 2.2 Kernels](https://github.com/huanghua1994/HiGP/blob/main/docs/2-Advanced-usage-of-HiGP.md#22-kernels). These kernel functions typically depend on one or more kernel parameters. For example, the Gaussian kernel $\kappa(x,y) = \exp(-\|x-y\|_2^2 / (2 l^2) )$ depends on the parameter $l$, typically known as the length-scale.

The quality of the model depends on the selection of the kernel function and the kernel parameters. HiGP focuses on optimizing kernel parameters and assumes that an appropriate kernel has been selected. To find the optimal $s$, $f$, and $l$ that best fit the data, an optimization process is generally required to minimize the negative log marginal likelihood:

```math
L(\Theta) = \frac{1}{2}\left(\mathbf{y}^{\top}\widehat{\mathbf{K}}^{-1}\mathbf{y} + \log|\widehat{\mathbf{K}}| + n\log 2\pi\right),
```

where $\widehat{\mathbf{K}}$ denotes the regularized kernel matrix $\kappa({\mathbf{X}},{\mathbf{X}}) + s \mathbf{I}$ and $\Theta$ denotes the hyperparameter set, which is $(s,f,l)$ for the above kernels. The optimization process usually requires the following gradient:

```math
\frac{\partial L}{\partial \theta} =
\frac{1}{2}{\mathbf{y}}^{\top}\widehat{\mathbf{K}}^{-1}\frac{\partial \widehat{\mathbf{K}}}{\partial \theta}\widehat{\mathbf{K}}^{-1}{\mathbf{y}} -
{\text{tr}}{(\widehat{\mathbf{K}}^{-1}\frac{\partial \widehat{\mathbf{K}}}{\partial \theta})},\quad \theta\in\Theta.
```

HiGP uses preconditioned Krylov subspace methods and preconditioned stochastic Lanczos quadrature trace estimators to efficiently compute $L(\Theta)$ and $\partial L / \partial \theta$ for all sizes of data sets.

## 2.2 Kernels

Currently, HiGP supports three commonly used kernels: Gaussian (RBF) kernel, Matern 3/2 kernel, and Matern 5/2 kernel.

In HiGP, the Gaussian kernel can be used by specifying `kernel_type=higp.GaussianKernel`. This kernel is defined as:

```math
K_{\text{Gaussian}}(x, y; l) = \exp \left( - \frac{\|x - y\|_2^2}{2 l^2} \right).
```

In HiGP, the Matern 3/2 kernel can be used by specifying `kernel_type=higp.Matern32Kernel`. This kernel is defined as:

```math
K_{\text{Matern32}}(x, y; l) = \left(1 + \sqrt{3} \frac{\|x - y\|_2}{l} \right) \exp \left( - \sqrt{3} \frac{\|x - y\|_2}{l} \right).
```

In HiGP, the Matern 5/2 kernel can be used by specifying `kernel_type=higp.Matern52Kernel`. This kernel is defined as:

```math
K_{\text{Matern52}}(x, y; l) = \left(1 + \sqrt{5} \frac{\|x - y\|_2}{l} + \frac{5 \|x - y\|_2^2}{3 l^2} \right) \exp \left( - \sqrt{5} \frac{\|x - y\|_2}{l} \right).
```

HiGP also supports defining a custom kernel. Please see [Section 2.6 Defining and using a custom kernel](https://github.com/huanghua1994/HiGP/blob/main/docs/2-Advanced-usage-of-HiGP.md#26-defining-and-using-a-custom-kernel) for detailed information.

## 2.3 Data structures

To set up and train a GP model or make predictions using a GP model, we need to provide a dataset to HiGP. For a dataset containing $N$ data points, with each data point having $d$ dimensions (for example, a point in a 3D space has $d=3$), the dataset should be stored as a 2D $d$-by-$N$ NumPy row-major (the default matrix / tensor storage style in NumPy) matrix, where each column stores one data point. For a dataset with 1D data points, it can either be an 1D array of length $N$, or a 2D matrix of shape $(1, N)$.

To set up and train a GP model, we also need to provide a label set. The label set should be an 1D NumPy array of length $N$.

Please use the `ascontiguousarray()` method in NumPy to ensure that the dataset matrix/array and the training label array are stored as a contiguous array.

## 2.4 Defining and training a GP model

Assume that we have prepared a training dataset `train_x` with shape $d$-by-$N$ and a training label set `train_y` of length $N$, both `train_x` and `train_y` are of type `np_dtype` (`np_dtype=numpy.float32` or `np_dtype=numpy.float64`). We can then define a GP regression model using the following four lines of code:

```python
torch_dtype = torch.float32 if np_dtype == numpy.float32 else torch.float64
gprproblem = higp.gprproblem.setup(data=train_x, label=train_y, kernel_type=higp.GaussianKernel)
model = higp.GPRModel(gprproblem, dtype=torch_dtype)
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
```

In this code listing:

* The first line defines the `torch_dtype` according to the `np_dtype` to ensure the GP training and PyTorch optimizer use the correct data type.
* The second line creates a `higp.gprproblem` object that contains the training data and specifies the kernel function to use (`kernel_type=higp.GaussianKernel` is the Gaussian kernel). This object computes of the model's error function and error function gradient at the current values of the hyperparameters.
* The third line creates a `GPRModel` object using the `higp.gprproblem` object. This object registers the hyperparameters as PyTorch parameters and set the gradients of PyTorch parameters in each step for the PyTorch optimizer.
* The forth line creates a [PyTorch Adam optimizer](https://pytorch.org/docs/stable/generated/torch.optim.Adam.html) with learning rate = 0.1 (`lr=0.1`). We can also use other optimizers in PyTorch.

The default initial pre-transformed (please see [Section 3.1 Definition of hyperparameters](https://github.com/huanghua1994/HiGP/blob/main/docs/3-API-reference.md#31-definition-of-hyperparameters) for the explanation of transform) hyperparameters $[l, f, s]$ are $[0, 0, 0]$. We can call `model.set_params()` with a NumPy 1D array of length 3 to manually set the pre-transformed hyperparameters.

After creating a GP model and a PyTorch optimizer, we can train the model using three lines of code:

```python
for i in ranges(maxits):
    loss = model.calc_loss_grad()   # Compute the loss and gradient
    optimizer.step()                # Update PyTorch parameters
```

We also provide a wrapper function `gpr_torch_minimize()` that does the same work and gathers and prints each iteration's loss and hyperparameter values.

After training the model, we can call `model.get_params()` to get the current (trained) hyperparameters. This function returns a NumPy 1D array of size 3, containing current pre-transformed hyperparameters $[l, f, s]$.

## 2.5 Make predictions

Assuming that we have a trained model `model` and a test dataset `test_x` of size $d$-by-$M$. We can make predictions with the trained model using:

```python
pred = higp.gpr_prediction(data_train=train_x,
                           label_train=train_y,
                           data_prediction=test_x,
                           kernel_type=higp.GaussianKernel,
                           pyparams=model.get_params())
```

`train_x`, `train_y`, and `kernel_type` are the same as what we used for calling `higp.gprproblem.setup()`, `data_prediction=test_x` specifies the prediction dataset, and `pyparams=model.get_params()` uses the trained pre-transformed hyperparameters for prediction. If we want to make a prediction with another set of hyperparameters, we simply provide a NumPy 1D array of length 3 containing pre-transformed hyperparameters $[l, f, s]$.

The returned structure `pred` has two members: `pred.prediction_mean` and `pred.prediction_stddev`, they are NumPy 1D arrays of length $M$. `prediction_mean` is the predicted mean values, and `prediction_stddev` is the standard deviation of prediction. For each prediction point `test_x[:, i]`, its prediction mean is `prediction_mean[i]`, and its 95\% prediction confidence range is `[prediction_mean[i] - 1.96 * prediction_stddev[i], prediction_mean[i] + 1.96 * prediction_stddev[i]]`.

## 2.6 Defining and using a custom kernel

Before proceeding in this section, please set a shell environment `REPOROOT` to the root directory of the HiGP repository.

To define and use your own kernel:

1. Edit `$REPOROOT/cpp-src/kernels/custom_kernel.cpp` to implement the kernel you want to use. Please read the "notes for implementing a custom kernel" at the beginning of this file before making modifications. If unmodified, it implements the Gaussian kernel as an example.
2. Rebuild the HiGP package on the local machine following the instructions below.
3. Uninstall the old HiGP package and install the local build of the HiGP package.
4. Use `kernel_type=higp.CustomKernel` in your Python code when calling related HiGP functions.

To use a custom kernel, you need to build the HiGP package on your local machine instead of installing the release versions using `pip`. We also strongly suggest building the HiGP Python package in a conda environment. To build the HiGP Python package, we recommend using:

* Python 3.10 or a newer version,
* NumPy 2.2.2 or a newer version.

You may use the following commands in bash to create a build environment using `conda`:

```bash
conda create -n py3.10 python=3.10 -y
conda activate py3.10
conda install -y libopenblas=*=*openmp*
conda install -y "blas=*=openblas"
conda install -y numpy=2.2.2
```

Run the following shell commands:

```shell
cd $REPOROOT/py-interface
python setup.py bdist_wheel
```

The packed wheel file will be in the `$REPOROOT/py-interface/dist` directory. This wheel file can be installed using `pip`.
