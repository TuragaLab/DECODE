import itertools as iter
import math
import numpy as np
import warnings
import os, sys
import torch
import matplotlib.pyplot as plt
import multiprocessing
from multiprocessing.dummy import Pool as ThreadPool

from deepsmlm.generic.emitter import EmitterSet, RandomEmitterSet
from deepsmlm.generic.noise import Poisson
from deepsmlm.generic.inout.load_calibration import SMAPSplineCoefficient
from deepsmlm.generic.phot_camera import Photon2Camera
from deepsmlm.generic.plotting.frame_coord import PlotFrame


class Simulation:
    """
    A class representing a SMLM simulation

    an image is represented according to pytorch convention, i.e.
    (N, C, H, W) in 2D - batched
    (C, H, W) in 2D - single image
    (N, C, D, H, W) in 3D - batched
    (C, D, H, W) in 2D - single image
    (https://pytorch.org/docs/stable/_modules/torch/nn/modules/conv.html#Conv3d)
    """
    def __init__(self, em, extent, psf, background, noise, frame_range=None, out_bg=False):
        """

        :param noise:
        :param em: set of emitters or sampler which outputs an emitterset upon calling
        :param extent: extent of the simulation, gives image-shape a meaning. tuple of tuples
        :param psf: instance of class psf
        :param background: instance of class background
        :param noise: instance of class noise
        :param frame_range: enforce the frame range of the simulation. None (default) / tuple
            usually we start with 0 (or the first frame) and end with the last frame, but you might want to have your
            simulation always contain a specific number of frames.
        :param out_bg: Output background additionally seperately? (true false)
        """

        self.em = em
        self.em_curr = None
        self.em_curr_split = None
        self.extent = extent
        self.frames = None
        self.frame_range = frame_range if frame_range is not None else (0, -1)

        self.psf = psf
        self.background = background
        self.noise = noise
        self.out_bg = out_bg

        """
        If the em input is a plain EmitterSet (and not a sampler) then just write it to the current emitterset 
        attribute
        """
        if isinstance(self.em, EmitterSet):
            self.em_curr = self.em

        """Split the Set of Emitters in frames. Outputs list of set of emitters."""
        if self.em_curr is not None:
            self.em_curr_split = self.em_curr.split_in_frames(self.frame_range[0], self.frame_range[1])

        if (self.background is not None) and (self.noise is None):
            """This is temporary since I changed the interface."""
            warnings.warn("Careful! You have not specified noise but you have specified background. "
                          "Background is defined as something which does not depend on the actual "
                          "signal whereas noise does.")

        if self.out_bg and (self.background is None):  # when we want to output bg we need to define it
            raise ValueError("If you want to output background in simulation you need to specify it. "
                             "Currently background was None.")

    @staticmethod
    def forward_single_frame(pos, phot, psf):
        """
        Render a single frame. Consist of psf forward and therafter bg forward.

        :param pos: torch tensor of xyz position
        :param phot: torch tensor of number of photons
        :param psf: psf instance
        :return: frame (torch.tensor)
        """
        frame = psf.forward(pos, phot)
        return frame

    def forward_single_frame_wrapper(self, emitter, psf):
        """
        Simple wrapper to unpack attributes of emitter set class.

        :param emitter: instance of class emitter set
        :param psf: instance of psf
        :return: returns a frame
        """
        pos = emitter.xyz
        phot = emitter.phot
        # id = emitter.id
        return self.forward_single_frame(pos, phot, psf)

    def forward(self, em_new=None):
        """
        Renders all frames.
        :param em_new: new set of emitters. Useful when forwarding a new emitter set every time you call this.
        :return: toch tensor of frames
        """

        if em_new is not None:
            self.em_curr = em_new
            self.em_curr_split = self.em_curr.split_in_frames(self.frame_range[0], self.frame_range[1])

        elif not isinstance(self.em, EmitterSet):  # do we have a sampler?
            self.em_curr = self.em()
            self.em_curr_split = self.em_curr.split_in_frames(self.frame_range[0], self.frame_range[1])

        em_sets = len(self.em_curr_split)
        frame_list = [None] * em_sets
        for i in range(em_sets):
            frame_list[i] = self.forward_single_frame_wrapper(self.em_curr_split[i],
                                                              self.psf)

        frames = torch.stack(frame_list, dim=0)

        """
        Add background. This needs to happen here and not on a single frame, since background may be correlated.
        The difference between background and noise is, 
        that background is assumed to be independent of the emitter position / signal.
        """
        if self.background is not None:
            bg_frames = self.background.forward(torch.zeros((frames.size(0), 1, frames.size(2), frames.size(3))))
            frames += bg_frames
        else:
            bg_frames = None

        if self.noise is not None:
            frames = self.noise.forward(frames)

        self.frames = frames
        return frames, bg_frames, self.em_curr

    def write_to_binary(self, outfile):
        raise NotImplementedError
