from typing import Union, Callable

import torch
from tqdm import tqdm

from .. import dataset
from ...generic import emitter
from functools import partial
import decode.utils


class Infer:

    def __init__(self, model, ch_in: int, frame_proc, post_proc, device: Union[str, torch.device],
                 batch_size: int = 64, num_workers: int = 4, pin_memory: bool = True,
                 forward_cat: Union[str, Callable] = 'emitter'):
        """
        Convenience class for inference.

        Args:
            model: pytorch model
            ch_in: number of input channels
            frame_proc: frame pre-procssing
            post_proc: post-processing
            device: device where to run inference
            batch_size: batch-size
            num_workers: number of workers
            pin_memory:
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

        self.forward_cat = self._setup_forward_cat(forward_cat)

    def forward(self, frames: torch.Tensor, sig_frames: torch.Tensor = None) -> emitter.EmitterSet:
        """
        Forward frames through model, pre- and post-processing and output EmitterSet

        Args:
            frames:

        """

        """Move Model"""
        model = self.model.to(self.device)
        model.eval()

        if model.sig_in:
            assert sig_frames is not None, 'Noise map has to be provided in addition to the frames'

        """Form Dataset and Dataloader"""
        if model.sig_in:
            if frames.shape == sig_frames.shape:
                ds = dataset.InferenceDataset(frames=frames, frame_proc=self.frame_proc, frame_window=self.ch_in, sig_frames=sig_frames)
            else:
                assert frames.shape[-2:] == sig_frames.shape[-2:], "Frames and noise map need to have the same image dimension"
        else:
            ds = dataset.InferenceDataset(frames=frames, frame_proc=self.frame_proc, frame_window=self.ch_in, sig_frames=None)

        dl = torch.utils.data.DataLoader(dataset=ds, batch_size=self.batch_size, shuffle=False,
                                         num_workers=self.num_workers, pin_memory=self.pin_memory,
                                         collate_fn=decode.neuralfitter.utils.collate.smlm_collate)

        out = []

        with torch.no_grad():
            for sample, sig_sample in tqdm(dl):
                if model.sig_in:
                    if sig_sample is not None:
                        sample = torch.cat([sample, sig_sample], 1)
                    elif sig_frames.ndim == 3:
                        sample = torch.cat([sample, sig_frames.unsqueeze(0).repeat(sample.shape[0], sample.shape[1], 1, 1)], 1)
                    elif sig_frames.ndim == 2:
                        sample = torch.cat([sample, sig_frames.unsqueeze(0).unsqueeze(0).repeat(sample.shape[0], sample.shape[1], 1, 1)], 1)

                x_in = sample.to(self.device)

                # compute output
                y_out = model(x_in)

                """In post processing we need to make sure that we get a single Emitterset for each batch, 
                so that we can easily concatenate."""
                out.append(self.post_proc.forward(y_out))

        """Cat to single emitterset / frame tensor depending on the specification of the forward_cat attr."""
        out = self.forward_cat(out)

        return out

    def _setup_forward_cat(self, forward_cat):

        if forward_cat is None:
            return lambda x: x

        elif isinstance(forward_cat, str):

            if forward_cat == 'emitter':
                return partial(emitter.EmitterSet.cat, step_frame_ix=self.batch_size)

            elif forward_cat == 'frames':
                return partial(torch.cat, dim=0)

        elif callable(forward_cat):
            return forward_cat

        else:
            raise TypeError(f"Specified forward cat method was wrong.")

        raise ValueError(f"Unsupported forward_cat value.")
