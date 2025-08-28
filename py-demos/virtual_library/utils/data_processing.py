"""Data processing utilities for Virtual Library."""

import numpy as np
from typing import Tuple


def train_test_normalize(
    train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray, test_y: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Normalize train and test sets using training statistics (Z-score normalization).

    This function applies z-score normalization to both features and labels,
    which is critical for numerical stability when dealing with synthetic functions
    that can have large output ranges (e.g., Rosenbrock function can reach 9000+).

    Args:
        train_x: Training features (N_train, D)
        train_y: Training labels (N_train,)
        test_x: Test features (N_test, D)
        test_y: Test labels (N_test,)

    Returns:
        Normalized train_x, train_y, test_x, test_y using training set statistics
    """
    mean_x = train_x.mean(axis=0, keepdims=True)
    std_x = (train_x.std(axis=0, keepdims=True) + 1e-6)
    train_x_norm = (train_x - mean_x) / std_x
    test_x_norm = (test_x - mean_x) / std_x
    mean_y = train_y.mean()
    std_y = train_y.std() + 1e-6
    train_y_norm = (train_y - mean_y) / std_y
    test_y_norm = (test_y - mean_y) / std_y
    return train_x_norm, train_y_norm, test_x_norm, test_y_norm


def denormalize_predictions(
    y_pred: np.ndarray, y_std: np.ndarray, y_mean: float, y_scale: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Denormalize predictions back to original scale.

    Args:
        y_pred: Normalized predictions
        y_std: Normalized prediction standard deviations
        y_mean: Original training label mean
        y_scale: Original training label standard deviation

    Returns:
        Denormalized predictions and standard deviations
    """
    y_pred_denorm = y_pred * y_scale + y_mean
    y_std_denorm = y_std * y_scale
    return y_pred_denorm, y_std_denorm
