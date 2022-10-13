from unittest import mock

import pytest

from decode.neuralfitter import logger
from pytorch_lightning.loggers.base import DummyLogger


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
