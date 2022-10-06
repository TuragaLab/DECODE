from abc import ABC
from pytorch_lightning import loggers


class Logger(loggers.LightningLoggerBase, ABC):
    # for the moment we just alias lightnings logger base
    # because the api seems reasonable
    pass
