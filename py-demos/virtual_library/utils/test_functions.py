import numpy as np


def rosenbrock(x, a=1.0, b=100.0):
    """Rosenbrock function (N-dimensional)

    f(x) = sum_{i=1}^{N-1} [b*(x_{i+1} - x_i^2)^2 + (a - x_i)^2]

    Domain: x_i ∈ [-2.048, 2.048]
    Global minimum: x = (1, 1, ..., 1), f(x) = 0
    """
    x = np.atleast_2d(x)
    if x.shape[0] == 1:
        x = x.T
    result = np.zeros(x.shape[0])
    for i in range(x.shape[0]):
        xi = x[i]
        val = 0.0
        for j in range(len(xi) - 1):
            val += b * (xi[j + 1] - xi[j] ** 2) ** 2 + (a - xi[j]) ** 2
        result[i] = val
    return result

def rastrigin(x, A=10.0):
    """Rastrigin function (N-dimensional)

    f(x) = A*n + sum_{i=1}^n [x_i^2 - A*cos(2*pi*x_i)]

    Domain: x_i ∈ [-5.12, 5.12]
    Global minimum: x = (0, 0, ..., 0), f(x) = 0
    """
    x = np.atleast_2d(x)
    if x.shape[0] == 1:
        x = x.T
    n = x.shape[1]
    return A * n + np.sum(x**2 - A * np.cos(2 * np.pi * x), axis=1)


def branin(x):
    """Branin function (2D only)

    f(x,y) = a*(y - b*x^2 + c*x - r)^2 + s*(1 - t)*cos(x) + s

    Domain: x ∈ [-5, 10], y ∈ [0, 15]
    Global minima: (-π, 12.275), (π, 2.275), (9.42478, 2.475), f(x*) ≈ 0.398
    """
    x = np.atleast_2d(x)
    if x.shape[0] == 1:
        x = x.T
    if x.shape[1] != 2:
        raise ValueError("Branin function is only defined for 2D")
    a = 1.0
    b = 5.1 / (4 * np.pi**2)
    c = 5.0 / np.pi
    r = 6.0
    s = 10.0
    t = 1.0 / (8 * np.pi)

    x1, x2 = x[:, 0], x[:, 1]
    term1 = a * (x2 - b * x1**2 + c * x1 - r) ** 2
    term2 = s * (1 - t) * np.cos(x1)

    return term1 + term2 + s

def get_function_info(func_name):
    """Get function info including bounds and optimal value"""
    info = {
        "rosenbrock": {
            "func": rosenbrock,
            "bounds": (-2.048, 2.048),
            "optimal": 0.0,
            "supports_dims": "any",
        },
        "rastrigin": {
            "func": rastrigin,
            "bounds": (-5.12, 5.12),
            "optimal": 0.0,
            "supports_dims": "any",
        },
        "branin": {
            "func": branin,
            "bounds": [(-5.0, 10.0), (0.0, 15.0)],
            "optimal": 0.397887,
            "supports_dims": 2,
        },
    }
    return info.get(func_name)
