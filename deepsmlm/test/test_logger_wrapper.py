import pytest

import matplotlib.pyplot as plt
import numpy as np

from deepsmlm.neuralfitter.utils import logger


class LoggerCheck:  # this does not have 'test' in its name because only the childs should be tested

    @pytest.fixture()
    def logger(self):
        raise NotImplementedError

    def test_op_scalar(self, logger):
        """
        Tests scalar logging

        Args:
            logger:

        """
        assert logger.add_scalar(42, 'the_number', 0)

    def test_op_plot(self, logger):
        """
        Tests plot logging

        Args:
            logger:

        """
        f = plt.figure()
        plt.plot(np.arange(10))

        assert logger.add_figure(f, 'the_plot', 0)

    def test_op_param(self, logger):
        """
        Tests param logging

        Args:
            logger:

        Returns:

        """
        assert logger.log_param(1e-4, 'dummy_scalar_param')
        assert logger.log_param({'a': 5, 'b': 10}, 'dummy_param_prefix')

    def test_no_op(self, logger):
        """Tests whether the logger returns None, which stands for no-op mode."""
        
        # make logger no-op
        logger.__init__(op=False)

        assert logger.add_scalar(42, 'the_number', 0) is None
        assert logger.add_figure(plt.figure(), 'the_plot', 0) is None
        assert logger.log_param(1e-4, 'dummy_scalar_param') is None
        assert logger.log_param({'a': 5, 'b': 10}, 'dummy_param_prefix') is None


class TestNoLog(LoggerCheck):

    @pytest.fixture()
    def logger(self):
        return logger.NoLog()

    def test_op_scalar(self, logger):
        assert logger.add_scalar(42, 'the_number', 0) is None

    def test_op_plot(self, logger):
        assert logger.add_figure(plt.figure(), 'the_plot', 0) is None

    def test_op_param(self, logger):
        assert logger.log_param(1e-4, 'dummy_scalar_param') is None
        assert logger.log_param({'a': 5, 'b': 10}, 'dummy_param_prefix') is None

    def test_no_op(self, logger):
        return
