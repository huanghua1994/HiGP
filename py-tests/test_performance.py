"""
Performance tests - verify model accuracy and error metrics meet expected thresholds
"""

import pytest
import numpy as np
import torch
import math
import higp


class TestPerformance:
    """Performance tests to verify HiGP model quality and convergence.

    These tests run sufficient iterations (50) to ensure models converge
    and verify that prediction accuracy meets expected thresholds.
    They validate both regression error (RMSE) and classification accuracy.
    """

    @pytest.mark.parametrize(
        "np_dtype,torch_dtype",
        [
            (np.float32, torch.float32),
            (np.float64, torch.float64),
        ],
    )
    def test_gpr_convergence_accuracy(self, np_dtype, torch_dtype):
        """
        Test GPR convergence and prediction accuracy after optimization.
        """
        np.random.seed(42)
        torch.manual_seed(42)
        n_train = 200
        n_test = 100
        train_x = np.linspace(0, 1, n_train).astype(np_dtype)
        train_y = 0.2 * np.exp(4 * train_x) + np.random.randn(n_train) * math.sqrt(0.09)
        train_y = train_y.astype(np_dtype)
        test_x = np.sort(np.random.rand(n_test)).astype(np_dtype)
        test_y = 0.2 * np.exp(4 * test_x) + np.random.randn(n_test) * math.sqrt(0.09)
        test_y = test_y.astype(np_dtype)
        gprproblem = higp.gprproblem.setup(
            data=train_x, label=train_y, kernel_type=higp.Matern32Kernel
        )
        model = higp.GPRModel(gprproblem, dtype=torch_dtype)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
        higp.gpr_torch_minimize(model, optimizer, maxits=50, print_info=False)
        pred = higp.gpr_prediction(
            data_train=train_x,
            label_train=train_y,
            data_prediction=test_x,
            kernel_type=higp.GaussianKernel,
            gp_params=model.get_params(),
        )
        rmse = np.linalg.norm(pred.prediction_mean - test_y) / np.sqrt(float(n_test))
        assert rmse < 0.3

    @pytest.mark.parametrize(
        "np_dtype,torch_dtype",
        [
            (np.float32, torch.float32),
            (np.float64, torch.float64),
        ],
    )
    def test_gpc_convergence_accuracy(self, np_dtype, torch_dtype):
        """
        Test GPC convergence and classification accuracy after optimization.
        """
        np.random.seed(42)
        torch.manual_seed(42)
        n_train = 200
        n_test = 100
        train_x = np.random.randn(2, n_train).astype(np_dtype)
        train_y = (train_x[0] + train_x[1] > 0).astype(np.int32)
        test_x = np.random.randn(2, n_test).astype(np_dtype)
        test_y = (test_x[0] + test_x[1] > 0).astype(np.int32)
        gpcproblem = higp.gpcproblem.setup(
            data=train_x, label=train_y, kernel_type=higp.GaussianKernel
        )
        model = higp.GPCModel(gpcproblem, num_classes=2, dtype=torch_dtype)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
        higp.gpc_torch_minimize(model, optimizer, maxits=50, print_info=False)
        pred = higp.gpc_prediction(
            data_train=train_x,
            label_train=train_y,
            data_prediction=test_x,
            kernel_type=higp.GaussianKernel,
            gp_params=model.get_params(),
        )
        acc = np.mean(pred.prediction_label == test_y)
        assert acc > 0.8

    @pytest.mark.parametrize(
        "np_dtype,torch_dtype",
        [
            (np.float32, torch.float32),
            (np.float64, torch.float64),
        ],
    )
    def test_ezgpr_torch_convergence(self, np_dtype, torch_dtype):
        """
        Test ezgpr_torch convergence and accuracy.
        """
        np.random.seed(42)
        torch.manual_seed(42)
        n_train = 200
        n_test = 100
        train_x = np.linspace(0, 1, n_train).astype(np_dtype)
        train_y = 0.2 * np.exp(4 * train_x) + np.random.randn(n_train) * math.sqrt(0.09)
        train_y = train_y.astype(np_dtype)
        test_x = np.sort(np.random.rand(n_test)).astype(np_dtype)
        test_y = 0.2 * np.exp(4 * test_x) + np.random.randn(n_test) * math.sqrt(0.09)
        test_y = test_y.astype(np_dtype)
        pred = higp.ezgpr_torch(
            train_x, train_y, test_x, test_y, adam_lr=0.1, adam_maxits=50
        )
        rmse = np.linalg.norm(pred.prediction_mean - test_y) / np.sqrt(float(n_test))
        assert rmse < 0.3

    @pytest.mark.parametrize(
        "np_dtype,torch_dtype",
        [
            (np.float32, torch.float32),
            (np.float64, torch.float64),
        ],
    )
    def test_ezgpc_torch_convergence(self, np_dtype, torch_dtype):
        """
        Test ezgpc_torch convergence and accuracy.
        """
        np.random.seed(42)
        torch.manual_seed(42)
        n_train = 200
        n_test = 100
        train_x = np.random.randn(2, n_train).astype(np_dtype)
        train_y = (train_x[0] + train_x[1] > 0).astype(np.int32)
        test_x = np.random.randn(2, n_test).astype(np_dtype)
        test_y = (test_x[0] + test_x[1] > 0).astype(np.int32)
        pred = higp.ezgpc_torch(
            train_x, train_y, test_x, test_y, adam_lr=0.1, adam_maxits=50
        )
        acc = np.mean(pred.prediction_label == test_y)
        assert acc > 0.8
