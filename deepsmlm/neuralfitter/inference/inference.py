from typing import Union

import torch
from tqdm import tqdm

from .. import dataset
from ...generic import emitter


class Infer:

    def __init__(self, model, ch_in, frame_proc, post_proc,
                 device: Union[str, torch.device], batch_size: int = 64):

        self.model = model
        self.ch_in = ch_in
        self.batch_size = batch_size
        self.device = device
        self.frame_proc = frame_proc
        self.post_proc = post_proc

    def forward(self, frames: torch.Tensor) -> emitter.EmitterSet:
        """
        Forward frames through model, pre- and post-processing and output EmitterSet

        Args:
            frames:

        """

        """Form Dataset and Dataloader"""
        ds = dataset.InferenceDataset(frames=frames, frame_proc=self.frame_proc, frame_window=self.ch_in)
        dl = torch.utils.data.DataLoader(dataset=ds, batch_size=self.batch_size, shuffle=False)

        """Move Model"""
        model = self.model.to(self.device)
        model.eval()

        """Eval mode."""
        em = []

        with torch.no_grad():
            for sample in tqdm(dl):
                x_in = sample.to(self.device)

                # compute output
                output = model(x_in)

                """In post processing we need to make sure that we get a single Emitterset for each batch, 
                so that we can easily concatenate."""
                em.append(self.post_proc.forward(output))

        em = emitter.EmitterSet.cat(em, step_frame_ix=self.batch_size)

        return em
