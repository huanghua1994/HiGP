"""
Smoke tests - verify basic functionality without errors
"""

import pytest
import numpy as np
import torch
import math
import higp


class TestSmoke:
    """Smoke tests to ensure basic HiGP interfaces work without errors.

    These tests run minimal iterations (2) to quickly verify that all
    main API functions can be called and return expected data structures.
    They do not verify accuracy or convergence quality.
    """

    @pytest.mark.parametrize(
        "np_dtype,torch_dtype",
        [
            (np.float32, torch.float32),
            (np.float64, torch.float64),
        ],
    )
    def test_gpr_basic_interface(self, np_dtype, torch_dtype):
        """
        Test that GPR (Gaussian Process Regression) basic interface works.
        """
        np.random.seed(42)
        torch.manual_seed(42)
        n_train = 200
        n_test = 100
        train_x = np.linspace(0, 1, n_train).astype(np_dtype)
        train_y = 0.2 * np.exp(4 * train_x) + np.random.randn(n_train) * math.sqrt(0.09)
        train_y = train_y.astype(np_dtype)
        test_x = np.sort(np.random.rand(n_test)).astype(np_dtype)
        gprproblem = higp.gprproblem.setup(
            data=train_x, label=train_y, kernel_type=higp.Matern32Kernel
        )
        model = higp.GPRModel(gprproblem, dtype=torch_dtype)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
        higp.gpr_torch_minimize(model, optimizer, maxits=2, print_info=False)
        pred = higp.gpr_prediction(
            data_train=train_x,
            label_train=train_y,
            data_prediction=test_x,
            kernel_type=higp.GaussianKernel,
            gp_params=model.get_params(),
        )

        assert pred is not None
        assert hasattr(pred, "prediction_mean")
        assert hasattr(pred, "prediction_stddev")
        assert pred.prediction_mean.shape == (n_test,)
        assert pred.prediction_stddev.shape == (n_test,)
        assert np.all(np.isfinite(pred.prediction_mean))
        assert np.all(np.isfinite(pred.prediction_stddev))
        assert np.all(pred.prediction_stddev > 0)

    @pytest.mark.parametrize(
        "np_dtype,torch_dtype",
        [
            (np.float32, torch.float32),
            (np.float64, torch.float64),
        ],
    )
    def test_gpc_basic_interface(self, np_dtype, torch_dtype):
        """
        Test that GPC (Gaussian Process Classification) basic interface works.
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
        higp.gpc_torch_minimize(model, optimizer, maxits=2, print_info=False)
        pred = higp.gpc_prediction(
            data_train=train_x,
            label_train=train_y,
            data_prediction=test_x,
            kernel_type=higp.GaussianKernel,
            gp_params=model.get_params(),
        )
        assert pred is not None
        assert hasattr(pred, "prediction_label")
        assert hasattr(pred, "prediction_probability")
        assert pred.prediction_label.shape == (n_test,)
        assert pred.prediction_probability.shape == (2, n_test)
        assert np.all(np.isin(pred.prediction_label, [0, 1]))
        assert np.all(pred.prediction_probability >= 0)
        assert np.all(pred.prediction_probability <= 1)
        assert np.allclose(pred.prediction_probability.sum(axis=0), 1.0, atol=1e-5)

    def test_kernel_types_available(self):
        """
        Test that all expected kernel types are available in the module.
        """
        assert higp.GaussianKernel is not None
        assert higp.Matern32Kernel is not None
        assert higp.Matern52Kernel is not None

    @pytest.mark.parametrize(
        "np_dtype,torch_dtype",
        [
            (np.float32, torch.float32),
            (np.float64, torch.float64),
        ],
    )
    def test_ezgpr_torch_interface(self, np_dtype, torch_dtype):
        """
        Test the ezgpr_torch GPR interface.
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
            train_x, train_y, test_x, test_y, adam_lr=0.1, adam_maxits=2
        )
        assert pred is not None
        assert hasattr(pred, "prediction_mean")
        assert hasattr(pred, "prediction_stddev")
        assert pred.prediction_mean.shape == (n_test,)
        assert pred.prediction_stddev.shape == (n_test,)
        assert np.all(np.isfinite(pred.prediction_mean))
        assert np.all(np.isfinite(pred.prediction_stddev))
        assert np.all(pred.prediction_stddev > 0)

    @pytest.mark.parametrize(
        "np_dtype,torch_dtype",
        [
            (np.float32, torch.float32),
            (np.float64, torch.float64),
        ],
    )
    def test_ezgpc_torch_interface(self, np_dtype, torch_dtype):
        """
        Test the ezgpc_torch GPC interface.
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
            train_x, train_y, test_x, test_y, adam_lr=0.1, adam_maxits=2
        )
        assert pred is not None
        assert hasattr(pred, "prediction_label")
        assert hasattr(pred, "prediction_probability")
        assert pred.prediction_label.shape == (n_test,)
        assert pred.prediction_probability.shape == (2, n_test)
        assert np.all(np.isin(pred.prediction_label, [0, 1]))
        assert np.all(pred.prediction_probability >= 0)
        assert np.all(pred.prediction_probability <= 1)
        assert np.allclose(pred.prediction_probability.sum(axis=0), 1.0, atol=1e-5)
