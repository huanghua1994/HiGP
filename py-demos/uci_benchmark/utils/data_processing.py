"""Data processing utilities for UCI benchmark."""

import numpy as np
from typing import Tuple, Optional


def train_test_normalize(
    train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray, test_y: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Normalize train and test sets using training statistics."""
    mean_x = train_x.mean(axis=0, keepdims=True)
    std_x = train_x.std(axis=0, keepdims=True) + 1e-6
    train_x_norm = (train_x - mean_x) / std_x
    test_x_norm = (test_x - mean_x) / std_x

    mean_y = train_y.mean()
    std_y = train_y.std() + 1e-6
    train_y_norm = (train_y - mean_y) / std_y
    test_y_norm = (test_y - mean_y) / std_y

    return train_x_norm, train_y_norm, test_x_norm, test_y_norm


def prepare_for_higp(
    X: np.ndarray, y: np.ndarray, dtype: Optional[np.dtype] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Prepare data for HiGP format (transpose X and make contiguous)."""
    if dtype is not None:
        X = X.astype(dtype)
        y = y.astype(dtype)

    # Transpose and make contiguous
    X_higp = np.ascontiguousarray(X.T)
    y_higp = np.ascontiguousarray(y)

    return X_higp, y_higp
