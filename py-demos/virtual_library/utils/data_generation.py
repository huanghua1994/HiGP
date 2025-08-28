import numpy as np
from scipy.stats import qmc


def latin_hypercube_sampling(n_samples, dim, bounds, seed=None):
    """Generate samples using Latin Hypercube Sampling

    Args:
        n_samples: Number of samples to generate
        dim: Dimension of the space
        bounds: Either a tuple (min, max) for all dimensions,
                or a list of tuples [(min1, max1), (min2, max2), ...]
        seed: Random seed for reproducibility

    Returns:
        Array of shape (n_samples, dim) with LHS samples
    """
    sampler = qmc.LatinHypercube(d=dim, seed=seed)
    samples = sampler.random(n=n_samples)
    if isinstance(bounds, tuple):
        min_val, max_val = bounds
        samples = samples * (max_val - min_val) + min_val
    else:
        for i, (min_val, max_val) in enumerate(bounds):
            samples[:, i] = samples[:, i] * (max_val - min_val) + min_val
    return samples


def add_noise(y, noise_level, seed=None):
    """Add Gaussian noise to labels

    Args:
        y: Labels
        noise_level: Standard deviation of noise (σ)
        seed: Random seed for reproducibility

    Returns:
        Noisy labels: y + ε, where ε ~ N(0, σ²)
    """
    if noise_level < 0:
        raise ValueError("noise_level must be non-negative")
    rng = np.random.RandomState(seed)
    noise = rng.normal(0, noise_level, size=y.shape)
    return y + noise


def generate_training_data(
    func, n_train, n_test, dim, bounds, noise_level=0.0, seed=None
):
    """Generate training and test data for a function

    Args:
        func: Test function to evaluate
        n_train: Number of training samples
        n_test: Number of test samples
        dim: Dimension of input space
        bounds: Function bounds
        noise_level: Relative noise level as fraction of data std (e.g., 0.01 = 1% noise)
        seed: Random seed

    Returns:
        train_x, train_y, test_x, test_y
    """
    if seed is not None:
        rng = np.random.RandomState(seed)
        train_seed = rng.randint(0, 2**31 - 1)
        test_seed = rng.randint(0, 2**31 - 1)
        noise_seed = rng.randint(0, 2**31 - 1)
    else:
        train_seed = None
        test_seed = None
        noise_seed = None
    train_x = latin_hypercube_sampling(n_train, dim, bounds, seed=train_seed)
    train_y = func(train_x)
    test_x = latin_hypercube_sampling(n_test, dim, bounds, seed=test_seed)
    test_y = func(test_x)
    if noise_level > 0:
        # Convert relative noise level to absolute noise based on data std
        data_std = train_y.std()
        absolute_noise_std = noise_level * data_std
        train_y = add_noise(train_y, absolute_noise_std, seed=noise_seed)
    return train_x, train_y, test_x, test_y
