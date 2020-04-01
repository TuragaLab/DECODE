import matplotlib.pyplot as plt

from abc import ABC, abstractmethod


class Logger(ABC):
    """Wraps a logger / multiple loggers and supports a no-op mode."""
    def __init__(self, op=True):
        super().__init__()

        self.op = op

    def add_scalar(self, val: (int, float), descr: str, ix: int):
        """
        Log a scalar. Calls the implementation method if not in no-op mode.

        Args:
            val (int,float): value to log
            descr (str): descriptor
            ix (int): index

         Returns:
            bool: if something was logged, None if not / no op

        """
        if self.op:
            self._log_scalar_impl(val, descr, ix)
            return True
            
    def add_figure(self, fig: plt.figure, descr: str, ix: int):
        """
        Log a figure. Calls the implemention method if not in no-op mode.

        Args:
            fig (plt.figure): figure to log
            descr (str): descriptor
            ix (int): index

        Returns:
            bool: if something was logged, None if not / no op

        """

        if self.op:
            self._log_plot_impl(fig, descr, ix)
            return True

    def log_param(self, param, descr):
        """
        Logs a parameter or a set of parameters (dictionary). Should only be called once per parameter

        Args:
            param:
            descr:

        Returns:
            bool: if something was logged, None if not / no op

        """
        if self.op:
            self._log_param_impl(param, descr)
            return True

    @abstractmethod
    def _log_scalar_impl(self, val, descr, ix):
        """
        Implementation of scalar logging.

        Args:
            val (int,float): value to log
            descr (str): descriptor
            ix (int): index

        """
        raise NotImplementedError

    @abstractmethod
    def _log_plot_impl(self, fig, descr, ix):
        """
        Implementation of plot logging.

        Args:
            fig (plt.figure): figure to log
            descr (str): descriptor
            ix (int): index

        """
        raise NotImplementedError

    @abstractmethod
    def _log_param_impl(self, param, descr):
        """
        Implementation of param logging

        Args:
            param: parameter to log. Can be numeric, string, dictionary
            descr: descriptor for single value logger, or prefix string for dict logging

        """
        raise NotImplementedError


class NoLog(Logger):
    def __init__(self):
        super().__init__(op=False)

    def _log_scalar_impl(self, val, descr, ix):
        return

    def _log_plot_impl(self, fig, descr, ix):
        return

    def _log_param_impl(self, param, descr):
        return
