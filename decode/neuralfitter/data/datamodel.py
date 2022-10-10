import pytorch_lightning as pl
import torch.utils.data
from torch.utils import data

from . import dataset
from . import experiment
from ..utils import dataloader_customs


class DataModel(pl.LightningDataModule):
    def __init__(
            self,
            experiment_train: experiment.Experiment,
            experiment_val: experiment.Experiment,
            num_workers: int
    ):
        super().__init__()

        self._exp_train = experiment_train
        self._exp_val = experiment_val
        self._num_workers = num_workers

        self._ds_val = None

    def prepare_data(self) -> None:
        # sample val set here, because we want to sample this only once
        x, y = self._exp_val.sample()
        self._ds_val = dataset.DatasetGausianMixture(x, y)

    def train_dataloader(self) -> data.DataLoader:
        # call this every epoch to get new data
        # i.e. `reload_dataloaders_every_n_epochs` hook in trainer
        x, y = self._exp_train.sample()

        ds = dataset.DatasetGausianMixture(x, y)
        return torch.utils.data.DataLoader(ds, num_workers=self._num_workers)

    def val_dataloader(self) -> data.DataLoader:
        return data.DataLoader(self._ds_val, num_workers=self._num_workers)
