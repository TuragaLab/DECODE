from unittest import mock

import matplotlib.pyplot as plt
import pytest
import torch
from pytorch_lightning.loggers.base import DummyLogger

from decode.neuralfitter import logger


def test_prefix_mixin():
    class MockLogger(logger.PrefixDictMixin, DummyLogger):
        def __init__(self):
            super().__init__()
            self._mock_log = mock.MagicMock()

        def log_metrics(self, *args, **kwargs):
            self._mock_log(*args, **kwargs)

    log = MockLogger()
    log.log_group({"a": 1}, prefix="loss/")
    log._mock_log.assert_called_once_with(metrics={"loss/a": 1}, step=None)


def test_log_tensor_mixin():
    class MockLogger(logger.LogTensorMixin, DummyLogger):
        def __init__(self):
            super().__init__()
            self._mock_log = mock.MagicMock()

        def log_figure(self, f, *args, **kwargs):
            self._mock_log(f, *args, **kwargs)
            plt.close(f)

    log = MockLogger()
    log.log_tensor(torch.rand(32, 32), "tensor", step=0)
    log._mock_log.assert_called_once()


@pytest.mark.parametrize("log_impl", [logger.TensorboardLogger])
def test_has_group_mixin(log_impl, tmpdir):
    log = log_impl(save_dir=tmpdir)
    log.log_group({"a": 1}, prefix="loss/")


@pytest.mark.parametrize("step_idx", [10, None])
def test_tensorboard_log_figure(tmpdir, step_idx):
    tb = logger.TensorboardLogger(tmpdir)
    tb.log_figure("dummy", plt.figure(), step_idx, close=True)  # functional test

    # test whether figure is closed etc.
    with mock.patch.object(tb.experiment, "add_figure") as mock_log:
        f = plt.figure()
        tb.log_figure("dummy", f, step_idx, close=True)

    mock_log.assert_called_once_with(
        tag="dummy", figure=f, global_step=step_idx, close=True
    )


def test_tensorboard_log_hist(tmpdir):
    tb = logger.TensorboardLogger(tmpdir)

    with mock.patch.object(tb.experiment, "add_histogram") as mock_hist:
        tb.log_hist("dummy_hist", torch.randn(100), step=0)

    mock_hist.assert_called_once()
