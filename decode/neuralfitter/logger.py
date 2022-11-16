from abc import ABC, abstractmethod
from typing import Optional

from pytorch_lightning import loggers


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


class Logger(PrefixDictMixin, loggers.LightningLoggerBase, ABC):
    # for the moment we just alias lightnings logger base
    # because the api seems reasonable
    pass


class TensorboardLogger(loggers.TensorBoardLogger, PrefixDictMixin):
    pass
