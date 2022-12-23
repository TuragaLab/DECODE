from abc import ABC, abstractmethod
from typing import Optional, Sequence, Union

import matplotlib.pyplot as plt
import torch
from pytorch_lightning import loggers
from pytorch_lightning.utilities import rank_zero_only


class PrefixDictMixin(ABC):
    @abstractmethod
    def log_metrics(
        self, metrics: dict[str, float], step: Optional[int] = None
    ) -> None:
        ...

    def log_group(
        self,
        metrics: dict[str, float],
        step: Optional[int] = None,
        prefix: Optional[str] = None,
    ) -> None:
        """

        Args:
            metrics: dictionary of metrics to log
            step:
            prefix: prefix to add before metric name
        """
        if prefix is not None:
            metrics = {prefix + k: v for k, v in metrics.items()}

        return self.log_metrics(metrics=metrics, step=step)


class LogTensorMixin(ABC):
    @abstractmethod
    def log_figure(self, figure: plt.figure, name: str, step: int, close: bool):
        ...

    def log_tensor(
        self,
        t: Union[torch.Tensor, Sequence[torch.Tensor]],
        name: str,
        step: Optional[int] = None,
        unbind: Optional[int] = None,
        colormap: str = "gray",
        colorbar: bool = True,
    ):
        t = t.detach().cpu()
        if unbind is not None:
            t = torch.unbind(t, dim=unbind)
        t = [t] if not isinstance(t, Sequence) else t

        for i, tt in enumerate(t):
            f, ax = plt.subplots()
            cax = ax.matshow(tt.numpy(), cmap=colormap)
            if colorbar:
                plt.colorbar(cax)
            self.log_figure(name=f"{name}/{i}", figure=f, step=step, close=True)


class Logger(PrefixDictMixin, loggers.LightningLoggerBase, ABC):
    def log_figure(
        self,
        name: str,
        figure: plt.figure,
        step: Optional[int] = None,
        close: bool = True,
    ) -> None:
        """
        Logs a matplotlib figure.
        Args:
            name: name of the figure
            figure: plt figure handle
            step: step number at which the figure should be recorded
            close: close figure after logging
        """
        if close:
            plt.close(figure)


class TensorboardLogger(loggers.TensorBoardLogger, LogTensorMixin, Logger):
    @rank_zero_only
    def log_figure(
        self,
        name: str,
        figure: plt.figure,
        step: Optional[int] = None,
        close: bool = True,
    ) -> None:
        self.experiment.add_figure(
            tag=name, figure=figure, global_step=step, close=close
        )
