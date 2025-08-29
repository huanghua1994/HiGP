"""Evaluation metrics."""

import numpy as np

MIN_STD = 1e-6


def compute_mse(y_true, y_pred):
    """Compute Mean Squared Error."""
    return np.mean((y_true - y_pred) ** 2)


def compute_rmse(y_true, y_pred):
    """Compute Root Mean Squared Error."""
    return np.sqrt(compute_mse(y_true, y_pred))


def compute_r2(y_true, y_pred):
    """Compute R-squared (coefficient of determination)."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0


def compute_nll(y_true, y_pred, y_std):
    """Compute Negative Log-Likelihood for Gaussian predictions.

    Args:
        y_true: True values
        y_pred: Predicted means
        y_std: Predicted standard deviations

    Returns:
        Average negative log-likelihood
    """
    y_std = np.maximum(y_std, MIN_STD)
    # NLL = 0.5 * log(2*pi*sigma^2) + 0.5 * ((y - mu) / sigma)^2
    nll = 0.5 * np.log(2 * np.pi * y_std**2) + 0.5 * ((y_true - y_pred) / y_std) ** 2
    return np.mean(nll)
