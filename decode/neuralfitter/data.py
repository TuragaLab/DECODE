import pytorch_lightning as pl
from torch.utils import data


class DataModel(pl.LightningDataModule):
    def __init__(self, ds_train: data.Dataset, ds_val: data.Dataset, num_workers: int):
        super().__init__()

        self._ds_train = ds_train
        self._ds_val = ds_val
        self._num_workers = num_workers

    def train_dataloader(self) -> data.DataLoader:
        return data.DataLoader(self._ds_train, num_workers=self._num_workers)

    def val_dataloader(self) -> data.DataLoader:
        return data.DataLoader(self._ds_val, num_workers=self._num_workers)
