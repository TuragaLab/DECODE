from typing import Union, Callable

import warnings
import torch
from tqdm import tqdm

from .. import dataset
from ...generic import emitter
from ...utils import hardware
from functools import partial


class Infer:

    def __init__(self, model, ch_in: int, frame_proc, post_proc, device: Union[str, torch.device],
                 batch_size: Union[int, str] = 'auto', num_workers: int = 4, pin_memory: bool = False,
                 forward_cat: Union[str, Callable] = 'emitter'):
        """
        Convenience class for inference.

        Args:
            model: pytorch model
            ch_in: number of input channels
            frame_proc: frame pre-processing pipeline
            post_proc: post-processing pipeline
            device: device where to run inference
            batch_size: batch-size or 'auto' if the batch size should be determined automatically (only use in combination with cuda)
            num_workers: number of workers
            pin_memory: pin memory in dataloader
            forward_cat: method which concatenates the output batches. Can be string or Callable.
            Use 'em' when the post-processor outputs an EmitterSet, or 'frames' when you don't use post-processing or if
            the post-processor outputs frames.
        """

        self.model = model
        self.ch_in = ch_in
        self.batch_size = batch_size
        self.device = device
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.frame_proc = frame_proc
        self.post_proc = post_proc

        self.forward_cat = None
        self._forward_cat_mode = forward_cat

        if str(self.device) == 'cpu' and self.batch_size == 'auto':
            warnings.warn("Automatically determining the batch size does not make sense on cpu device. "
                          "Falling back to reasonable value.")
            self.batch_size = 64

    def forward(self, frames: torch.Tensor) -> emitter.EmitterSet:
        """
        Forward frames through model, pre- and post-processing and output EmitterSet

        Args:
            frames:

        """

        """Move Model"""
        model = self.model.to(self.device)
        model.eval()

        """Form Dataset and Dataloader"""
        ds = dataset.InferenceDataset(frames=frames, frame_proc=self.frame_proc, frame_window=self.ch_in)
        
        if self.batch_size == 'auto':
            # include safety factor of 20%
            bs = int(0.8 * self.get_max_batch_size(model, ds[0].size(), 1, 512))
        else:
            bs = self.batch_size
            
        # generate concatenate function here because we need batch size for this
        self.forward_cat = self._setup_forward_cat(self._forward_cat_mode, bs)

        dl = torch.utils.data.DataLoader(dataset=ds, batch_size=bs, shuffle=False, drop_last=False,
                                         num_workers=self.num_workers, pin_memory=self.pin_memory)

        out = []

        with torch.no_grad():
            for sample in tqdm(dl):
                x_in = sample.to(self.device)

                # compute output
                y_out = model(x_in)

                """In post processing we need to make sure that we get a single Emitterset for each batch, 
                so that we can easily concatenate."""
                out.append(self.post_proc.forward(y_out))

        """Cat to single emitterset / frame tensor depending on the specification of the forward_cat attr."""
        out = self.forward_cat(out)

        return out

    def _setup_forward_cat(self, forward_cat, batch_size: int):

        if forward_cat is None:
            return lambda x: x

        elif isinstance(forward_cat, str):

            if forward_cat == 'emitter':
                return partial(emitter.EmitterSet.cat, step_frame_ix=batch_size)

            elif forward_cat == 'frames':
                return partial(torch.cat, dim=0)

        elif callable(forward_cat):
            return forward_cat

        else:
            raise TypeError(f"Specified forward cat method was wrong.")

        raise ValueError(f"Unsupported forward_cat value.")

    def get_max_batch_size(self, model: torch.nn.Module, frame_size: Union[tuple, torch.Size], limit_low: int, limit_high: int):
        """
        Get maximum batch size for inference.

        Args: 
            model: model on correct device
            frame_size: size of frames (without batch dimension)
        """
        def model_forward_no_grad(x: torch.Tensor):
            """
            Helper function because we need to account for torch.no_grad()
            """
            with torch.no_grad():
                o = model.forward(x)
            
            return o

        assert next(model.parameters()).is_cuda, "Auto determining the max batch size makes only sense when running on CUDA device."
    
        return hardware.get_max_batch_size(model_forward_no_grad, frame_size, next(model.parameters()).device, limit_low, limit_high)

