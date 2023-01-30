from typing import Any, Optional, Protocol

import pytorch_lightning as pl
import torch

from decode.emitter import emitter
from decode.neuralfitter import logger, process
from decode.evaluation import predict_dist


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
    def logger(self) -> Optional[logger.TensorboardLogger]:
        return super().logger

    def configure_optimizers(self):
        return self._opt

    def transfer_batch_to_device(self, batch: Any, device: torch.device, dataloader_idx: int) -> Any:
        if len(batch) == 3:
            return super().transfer_batch_to_device(batch[:-1], device, dataloader_idx) + (batch[-1], )
        return super().transfer_batch_to_device(batch, device, dataloader_idx)

    def training_step(self, batch, batch_ix: int):
        opt = self.optimizers()
        opt.zero_grad()

        x, y = batch

        y_raw = self._model.forward(x)
        y_post = self._proc.post_model(y_raw)
        loss, loggable = self._loss.forward(y_post, y)

        self.manual_backward(loss)
        torch.nn.utils.clip_grad_norm_(
            self._model.parameters(), max_norm=0.03, norm_type=2
        )
        opt.step()

        self.log(
            "loss/train", loss, on_step=True, on_epoch=True, batch_size=self.batch_size
        )
        self.logger.log_group(loggable, prefix="loss/train_", step=self.global_step)

        if batch_ix == 0:  # graphic logging
            ix = 0
            self.logger.log_tensor(
                x[ix], name="input_train", step=self.global_step, unbind=0
            )
            self.logger.log_tensor(
                y_raw[ix], name="output_raw_train", step=self.global_step, unbind=0
            )
            self.logger.log_tensor(
                y_post[ix], name="output_post_model_train", step=self.current_epoch,
                unbind=0
            )

        return loss

    def validation_step(self, batch, batch_ix: int):
        x, y_ref, y_em_val = batch
        # convert em dict back to em set


        y_raw = self._model.forward(x)
        y_post = self._proc.post_model(y_raw)
        loss, loggable = self._loss.forward(y_post, y_ref)

        self.log(
            "loss/val", loss, on_step=False, on_epoch=True, batch_size=self.batch_size
        )
        # self.logger.log_group(loggable, prefix="loss_cmpt/val", )

        y_flip = y_raw.clone()
        em_out = self._proc.post(y_flip)
        em_out.frame_ix += batch_ix * self.batch_size

        self._em_val_out.append(em_out)
        self._em_val_tar.extend(y_em_val)

        if batch_ix == 0:  # graphic logging
            ix = 0  # index in batch

            em_out_log = em_out.iframe[ix]

            self.logger.log_tensor(
                x[ix], name="input_val", step=self.current_epoch, unbind=0
            )
            self.logger.log_emitter(
                name="input_em_val",
                em_tar=y_em_val[ix],
                frame=x[ix, 1],
                step=self.current_epoch,
            )
            self.logger.log_emitter(
                name="output_em_val/p010",
                em_tar=y_em_val[ix],
                em=em_out_log[em_out_log.prob >= 0.1],
                frame=x[ix, 1],
                step=self.current_epoch,
            )
            self.logger.log_emitter(
                name="output_em_val/p050",
                em_tar=y_em_val[ix],
                em=em_out_log[em_out_log.prob >= 0.5],
                frame=x[ix, 1],
                step=self.current_epoch,
            )
            self.logger.log_emitter(
                name="output_em_val/p090",
                em_tar=y_em_val[ix],
                em=em_out_log[em_out_log.prob >= 0.9],
                frame=x[ix, 1],
                step=self.current_epoch,
            )
            self.logger.log_tensor(
                y_raw[ix], name="output_raw_val", step=self.current_epoch, unbind=0
            )
            self.logger.log_tensor(
                y_post[ix], name="output_post_model_val", step=self.current_epoch,
                unbind=0
            )
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
        self.logger.log_group(
            {"n_out": len(em_out), "n_tar": len(em_tar)},
            prefix="eval/",
            step=self.current_epoch,
        )

        self.logger.log_hist(
            name="tar_em_dist_val/frame_ix",
            vals=em_tar.frame_ix,
            step=self.current_epoch,
        )

        if len(em_out) >= 1:
            self.logger.log_hist(
                name="output_em_dist_val/prob",
                vals=em_out.prob,
                step=self.current_epoch,
            )
            self.logger.log_hist(
                name="output_em_dist_val/frame_ix",
                vals=em_out.frame_ix,
                step=self.current_epoch,
            )
            self.logger.log_hist(
                name="output_em_dist_val/x",
                vals=em_out.xyz[:, 0],
                step=self.current_epoch,
            )
            self.logger.log_hist(
                name="output_em_dist_val/y",
                vals=em_out.xyz[:, 1],
                step=self.current_epoch,
            )
            self.logger.log_hist(
                name="output_em_dist_val/z",
                vals=em_out.xyz[:, 2],
                step=self.current_epoch,
            )
            self.logger.log_hist(
                name="output_em_dist_val/x_offset",
                vals=predict_dist.px_pointer_dist(em_out.xyz[:, 0], -0.5, 1.),
                step=self.current_epoch,
            )
            self.logger.log_hist(
                name="output_em_dist_val/y_offset",
                vals=predict_dist.px_pointer_dist(em_out.xyz[:, 1], -0.5, 1.),
                step=self.current_epoch,
            )

        # ToDo: emitter based distributions
        # ToDo: graphical samples
