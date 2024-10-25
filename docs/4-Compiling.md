# 4. Compiling

Before proceeding in this section, please set a shell environment `REPOROOT` to the root directory of the HiGP repository.

## 4.1 Structure of HiGP

HiGP is a Python 3 library with C++ extensions. The performance-critical parts have been implemented to achieve very high performance. This subsection introduces the structure of HiGP.

The HiGP C++ part implements five functional units:

1. `kernel` unit, in `$REPOROOT\cpp-src\kernels`. This unit computes two matrices $K(X, Y; l)$ and $\partial K(X, Y; l) / \partial l$ for two sets of points $X$ and $Y$ and a lengthscale parameter $l$. Currently, the pairwise squared distance kernel (`pdist2`), Gaussian (RBF) kernel (`gaussian`), Matern 3/2 and 5/2 kernels (`matern32` and `matern52`). The `pdist2` kernel does not have a lengthscale parameter $l$, and it is used by other kernels.
2. `h2mat` unit, in `$REPOROOT\cpp-src\h2mat`. This unit relies on the `kernel` unit. This unit computes the $\mathcal{H}^2$ matrix representations of two matrices $K(X, Y; l)$ and $\partial K(X, Y; l) / \partial l$. The wrapper class `ss_h2mat` computes the scaled and shifted kernel matrix $\hat{K} = f^2 K(X, Y; l) + s I$ for hyperparameters $[l, f, s]$ and the multiplication of $\partial \hat{K} / \partial \theta \times B$ for $\theta \in \{l, f, s\}$ and and dense general matrix $B$.
3. `dense_krnl_mat` unit, in `$REPOROOT\cpp-src\dense_kernel_matrix.{h, cpp}`. This unit relies on the `kernel` unit. This unit computes the scaled and shifted kernel matrix $\hat{K} = f^2 K(X, Y; l) + s I$ for hyperparameters $[l, f, s]$ and the multiplication of $\partial \hat{K} / \partial \theta \times B$ for $\theta \in \{l, f, s\}$ and dense general matrix $B$.
4. `solver` unit, in `$REPOROOT\cpp-src\solvers`. This unit relies on the `h2mat` and `dense_krnl_mat` units. This unit implements a blocked preconditioned conjugate gradient (block PCG) iterative solver and the adaptive factorized Nystrom (AFN) preconditioner.
5. `gp` unit, in `$REPOROOT\cpp-src\gp`. This unit relies on the `solver`, `h2mat`, and `dense_krnl_mat` units. This unit computes the loss and gradients for GP regression and GP classification. This unit includes two sub-units: `exact_gp` and `precond_gp`. The `exact_gp` sub-unit implements the GP computations using dense matrix factorizations and exact solves. The `precond_gp` sub-unit implements the GP computations using iterative solvers and the stochastic Lanczos quadrature method.

The aforementioned C++ units are wrapped into four basic Python modules:

1. `higp.krnlmatmodule` module. This module wraps and calls the C++ `h2mat` and `dense_krnl_mat` modules.
2. `higp.precondmodule` module. This module wraps and calls the AFN precondtioner in the C++ `solver` module.
3. `higp.gprproblemmodule` module. This module calls the C++ `gp` module to compute the loss and gradient for GP regression.
4. `higp.gpcproblemmodule` module. This module calls the C++ `gp` module to compute the loss and gradient for GP classification.

The HiGP Python part implements the following modules:

1. `higp.GPRModel` module. This module registers the hyperparameters used in GP regression as PyTorch parameters and set the gradients of PyTorch parameters in each step for the PyTorch optimizer.
2. `higp.GPCModel` module. This module registers the hyperparameters used in GP classification as PyTorch parameters and set the gradients of PyTorch parameters in each step for the PyTorch optimizer.

## 4.2 Building HiGP on your local machine

For testing or debugging purposes, you may want to build the HiGP package on your local machine instead of installing the release versions using `pip`. We also strongly suggest building the HiGP Python package in a conda environment. To build the HiGP Python package, you need:

* Python 3.7 or a newer version,
* NumPy 1.15 or a newer version.

Run the following shell commands:

```shell
cd $REPOROOT/py-interface
python setup.py bdist_wheel
```

The packed wheel file will be in the `$REPOROOT/py-interface/dist` directory. This wheel file can be installed using `pip`.

## 4.3 Travis CI

This repository has a Travis CI configuration file `.travis.yml` for two tasks:

1. Run the `$REPOROOT/travis/cpp-unit-tests.sh` script for C++ unit tests (if any test fails, Travis CI will interrupt and report a failed build),
2. Run the `$REPOROOT/travis/build-wheels.sh` script for building the Python packages for release (but will not automatically push them to PyPI).

The release packages are built in the [pypa manylinux2014](https://github.com/pypa/manylinux) environment and targets CPUs with the AVX-2 instruction sets for best compatibility.

To obtain the manylinux2014 docker image, run this shell command:

```shell
sudo docker pull quay.io/pypa/manylinux2014_x86_64
```

To enter the docker image for testing, run the following shell commands:

```shell
cd $REPOROOT
sudo docker run -it --rm -v `pwd`:/io quay.io/pypa/manylinux2014_x86_64 bash
```

This command mounts the root directory of this repository to the `/io/` directory in the docker image and starts the bash shell.

## 4.4 Building the Python packages for release

Run the following shell commands:

```shell
cd $REPOROOT
sudo docker run --rm -v `pwd`:/io quay.io/pypa/manylinux2014_x86_64 /io/travis/build-wheels.sh
```

The built wheel package files will be in the root directory of this repository.

## 4.5 Debugging the C/C++ source code on local machine

You need a C compiler and a C++ compiler mentioned before and a BLAS+LAPACK library (we have tested MKL and OpenBLAS).

To compile the C++ library and the C++ unit test codes, first modify the `$REPOROOT/makefile.in` file in the root directory of this repository based on the compiler and the BLAS+LAPACK library you are using. We provide two example files:

* `$REPOROOT/makefile-gcc-openblas.in` for GCC + OpenBLAS,
* `$REPOROOT/makefile-icc-mkl.in` for classic ICC + MKL (ICC 2020 or earlier, doesn't work with the new Intel oneAPI compilers).

Then, enter directory `$REPOROOT/cpp-src` and run `make`. This will build the C++ library and the built library are in `$REPOROOT/cpp-lib`.

Then, enter directory `$REPOROOT/cpp-tests` and run `make`. This will build the C++ unit test programs.

To run the unit test programs, see the last section in `$REPOROOT/travis/cpp-unit-tests.sh` for detailed commands.
