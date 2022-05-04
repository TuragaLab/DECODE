import warnings
import pytest
import torch

from decode.neuralfitter.utils import logger


def test_noop_logger():
    logger.NoLog()
    warnings.warn("No proper test implemented.")


class TestDictLogger:

    @pytest.fixture()
    def logger(self):
        return logger.DictLogger()

    def test_log_scalar(self, logger):
        scalar_data = [5., -1., 2., 3]

        for i, s in enumerate(scalar_data):
            logger.add_scalar("dummy_scalar", s, i)

        """Assert"""
        assert (torch.tensor(logger.log_dict["dummy_scalar"]['scalar']) == torch.tensor(scalar_data)).all()


class TestMultiLogger:
    class ALogger:
        def log_scalar(self, a):
            # print(f"{a}")
            return a

    class BLogger:
        def log_scalar(self, a):
            # print(f"{2*a}")
            return 2 * a

    @pytest.fixture()
    def logger(self):
        return logger.MultiLogger([self.ALogger(), self.BLogger()])

    def test_arbitrary_execution(self, logger):
        out = logger.log_scalar(5.)

        assert (torch.tensor(out) == torch.tensor([5., 10.])).all()
