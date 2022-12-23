from unittest import mock

import matplotlib.pyplot as plt
import pytest
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
