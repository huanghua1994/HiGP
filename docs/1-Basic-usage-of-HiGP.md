# 1. Basic usage of HiGP

## 1.1 Install HiGP

HiGP is a Python3 library that can be installed using `pip`. We strongly recommend using HiGP in a [conda](https://www.anaconda.com/blog/understanding-conda-and-pip) environment. You may follow [this page](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) to install conda and create a conda environment. To install HiGP, make sure you have installed the following packages:

* Python 3.7 or a newer version,
* NumPy 1.15 or a newer version,
* SciPy 1.3 or a newer version,
* PyTorch 1.3 or a newer version.

Then, simply run the following command in your conda environment to download and install the latest version of HiGP.

```python
pip install HiGP
```

## 1.2 A simple example

We use a simple example to show the basic usage of HiGP. In this example, we train a RBF (Gaussian) kernel GP for a one-dimensional function. We will be modeling the function

```math
y = 0.2 \sin(3 \pi x) \exp(4x) + \epsilon, \ 
\epsilon \sim \mathcal{N}(0, 0.09)
```

with 200 training examples, and testing on 1000 examples.

First import the libraries we need to use:

```python
import numpy as np
import matplotlib.pyplot as plt
import math
import higp
%matplotlib inline
```

Now let's set up the training and testing data. We use 200 regularly spaced points on $[0, 1]$ which we evaluate the function on and add Gaussian noise to get the training labels.

```python
np.random.seed(42)
n_train = 200
n_test  = 1000
train_x = np.linspace(0, 1, n_train)
train_y = 0.2 * np.sin(3 * math.pi * train_x) * np.exp(4 * train_x) 
train_y += np.random.randn(train_x.size) * math.sqrt(0.09)
test_x  = np.sort(np.random.rand(n_test))
test_y  = 0.2 * np.sin(3 * math.pi * test_x) * np.exp(4 * test_x)
test_y += np.random.randn(test_x.size) * math.sqrt(0.09)
```

The following figure shows the training set and the test set:

![Example00 - training sets and test sets](figs/Example00_1.png)

HiGP has an "easy GP regression" (ezgpr) interface to train the model and test the trained model in one line:

```python
pred_y, std_y = higp.ezgpr_torch(train_x, train_y, test_x, test_y, adam_lr=0.1, adam_maxits=100)
```

This "ezgpr" interface uses the RBF (Gaussian) kernel, [PyTorch Adam optimizer](https://pytorch.org/docs/stable/generated/torch.optim.Adam.html) in the training with a learning rate 0.1 (`adam_lr=0.1`) and 100 training iterations (`adam_maxits=100`). It returns two arrays of the same size as `test_x`: `pred_y` is the prediction mean values, `std_y` is the prediction standard deviations.

We can use the following code to visualize the prediction:

```python
plt.figure(figsize=(5, 5))
plt.plot(test_x, test_y, 'ro', markersize=2)
plt.plot(test_x, pred_y, 'b-')
plt.fill_between(test_x, pred_y - 1.96 * std_y, pred_y + 1.96 * std_y, alpha=0.5)
plt.title('Testing label and prediction')
plt.xlabel('x')
plt.ylabel('y')
plt.legend(['Testing label', 'Prediction Mean', '95% Confidence'], loc='upper center')
plt.show()
plt.close(fig)
```

![Example00 - prediction](figs/Example00_2.png)

Instead of using the "ezgpr" interface, we can also manually define a model for this problem. Before defining the model, we first need to make sure the training and testing data are of the same data type:

```python
np_dtype = np.float32
torch_dtype = torch.float32 if np_dtype == np.float32 else torch.float64
train_x = train_x.astype(np_dtype)
train_y = train_y.astype(np_dtype)
test_x = test_x.astype(np_dtype)
test_y = test_y.astype(np_dtype)
```

Then we can set up a GP regression model in HiGP:

```python
gpr_problem = higp.gprproblem.setup(data=train_x, label=train_y, kernel_type=1)
model = higp.GPRModel(gpr_problem, dtype=torch_dtype)
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
```

Then, train the model:

```python
higp.gpr_torch_minimize(model, optimizer, maxits=100, print_info=True)
```

using 50 training iterations (`maxits=50`) and print the training information (`print_info=True`).

Finally, we can do prediction with the trained model:

```python
pred = higp.gpr_prediction(data_train=train_x,
                           label_train=train_y,
                           data_prediction=test_x,
                           kernel_type=1,
                           pyparams=model.get_np_params())
pred_y = pred[0]
std_y = pred[1]
```

The obtained `pred_y` and `std_y` are the same as the outputs of `higp.ezgpr_torch()`.