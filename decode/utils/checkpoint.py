import torch
import pytest

from pathlib import Path
from typing import Union


class CheckPoint:
    def __init__(self, path: Union[str, Path]):
        self.path = path

        self.model_state = None
        self.optimizer_state = None
        self.lr_sched_state = None
        self.step = None
        self.log = None

    @property
    def dict(self):
        return {
            'step': self.step,
            'model_state': self.model_state,
            'optimizer_state': self.optimizer_state,
            'lr_sched_state': self.lr_sched_state,
            'log': self.log
        }

    def update(self, model_state, optimizer_state, lr_sched_state, step, log):
        self.model_state = model_state
        self.optimizer_state = optimizer_state
        self.lr_sched_state = lr_sched_state
        self.step = step
        self.log = log

    def save(self):
        torch.save(self.dict, self.path)

    def dump(self, model_state, optimizer_state, lr_sched_state, step, log=None):
        self.update(model_state, optimizer_state, lr_sched_state, step, log)
        self.save()
