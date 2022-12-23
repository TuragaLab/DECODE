from typing import Optional, Protocol

import pytorch_lightning as pl
import torch

from decode.emitter import emitter
from decode.neuralfitter import logger, process


class _EvaluatorEmitter(Protocol):
    def forward(self, out: emitter.EmitterSet, ref: emitter.EmitterSet) -> dict:
        pass


class Model(pl.LightningModule):
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        lr_sched: torch.optim.lr_scheduler.StepLR,
        loss: torch.nn.Module,
        proc: process.ProcessingSupervised,
        evaluator: Optional[_EvaluatorEmitter],
        batch_size: int,
    ):
        super().__init__()
        self.automatic_optimization = False

        self._model = model
        self._opt = optimizer
        self._lr_sched = lr_sched
        self._proc = proc
        self._loss = loss
        self._evaluator = evaluator
        self.batch_size = batch_size
        self._em_val_out = []
        self._em_val_tar = []

    @property
    def logger(self) -> Optional[logger.Logger]:
        return super().logger

    def configure_optimizers(self):
        return self._opt

    def training_step(self, batch, batch_ix: int):
        x, y = batch

        y_out = self._model.forward(x)
        y_out = self._proc.post_model(y_out)
        loss, loggable = self._loss.forward(y_out, y)

        self.log(
            "loss/train", loss, on_step=True, on_epoch=True, batch_size=self.batch_size
        )
        self.logger.log_group(loggable, prefix="loss_cmpt/")

        return loss

    def validation_step(self, batch, batch_ix: int):
        x, y_ref, y_em_val = batch

        y_out = self._model.forward(x)
        y_out_proc = self._proc.post_model(y_out)
        loss, loggable = self._loss.forward(y_out_proc, y_ref)

        self.log(
            "loss/val", loss, on_step=True, on_epoch=True, batch_size=self.batch_size
        )
        self.logger.log_group(loggable, prefix="loss_cmpt/")

        em_out = self._proc.post(y_out)
        em_out.frame_ix += batch_ix * self.batch_size

        self._em_val_out.append(em_out)
        self._em_val_tar.extend(y_em_val)

        return loss

    def on_train_epoch_end(self) -> None:
        self._lr_sched.step()

    def on_validation_epoch_end(self) -> None:
        if self._evaluator is None:
            return

        em_out = emitter.EmitterSet.cat(self._em_val_out)
        em_tar = emitter.EmitterSet.cat(self._em_val_tar)

        # emitter based metrics
        metrics = self._evaluator.forward(em_out, em_tar)
        self.logger.log_group(metrics, prefix="eval/", step=self.current_epoch)

        # ToDo: emitter based distributions
        # ToDo: graphical samples
