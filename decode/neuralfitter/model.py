from typing import Optional

import pytorch_lightning as pl
import torch


from decode.emitter import emitter


class Model(pl.LightningModule):
    def __init__(
            self,
            model: torch.nn.Module,
            loss: torch.nn.Module,
            proc,
            em_val_tar: emitter.EmitterSet,
            evaluator: Optional,
    ):
        super().__init__()

        self._model = model
        self._proc = proc
        self._loss = loss
        self._em_val_out = []
        self._em_val_tar = em_val_tar
        self._evaluator = evaluator

    def training_step(self, batch, batch_ix: int):
        x, y = batch

        y_out = self._model.forward(x)
        y_out = self._proc.post_model(y_out)
        loss = self._loss.forward(y, y_out)

        self.log("loss/train", loss, on_step=True, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_ix: int):
        x, y = batch

        y_out = self._model.forward(x)
        y_out_proc = self._proc.post_model(y_out)
        loss = self._loss.forward(y, y_out_proc)

        em_out = self._proc.post(y_out)
        em_out.frame_ix += batch_ix  # correct for batching
        self._em_val_out.append(em_out)

        self.log("loss/val", loss, on_step=True, on_epoch=True)

        return loss

    def on_validation_epoch_end(self) -> None:
        if self._evaluator is None:
            return

        metrics = self._evaluator(self._em_val_tar, self._em_val_out)
        self.log_dict(metrics, on_epoch=True)
