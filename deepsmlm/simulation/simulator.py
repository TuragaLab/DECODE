import itertools as iter
import math
import numpy as np
import os, sys
import torch
import multiprocessing
from multiprocessing.dummy import Pool as ThreadPool

from deepsmlm.generic.emitter import EmitterSet
from deepsmlm.generic.noise import Poisson
from deepsmlm.generic.inout.load_calibration import SMAPSplineCoefficient


class Simulation:
    """
    A class representing a smlm simulation

    an image is represented according to pytorch convention, i.e.
    (N, C, H, W) in 2D - batched
    (C, H, W) in 2D - single image
    (N, C, D, H, W) in 3D - batched
    (C, D, H, W) in 2D - single image
    (https://pytorch.org/docs/stable/_modules/torch/nn/modules/conv.html#Conv3d)
    """
    def __init__(self, em, extent, psf, background=None, poolsize=4, frame_range=None):
        """

        :param em: set of emitters. instance of class EmitterSet
        :param extent: extent of the simulation, gives image-shape a meaning. tuple of tuples
        :param psf: instance of class psf
        :param background: instance pf class background / noise
        :param poolsize: how many threads should be open to calculate psf
        :param frame_range: enforce the frame range of the simulation. None (default) / tuple
            usually we start with 0 (or the first frame) and end with the last frame, but you might want to have your
            simulation always contain a specific number of frames.
        """

        self.em = em
        self.em_split = None
        self.extent = extent
        self.frames = None
        self.frame_range = frame_range if frame_range is not None else (0, -1)

        self.psf = psf
        self.background = background
        self.poolsize = poolsize

        """Split the Set of Emitters in frames. Outputs list of set of emitters."""
        if self.em is not None:
            self.em_split = self.em.split_in_frames(self.frame_range[0], self.frame_range[1])

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
            self.em = em_new
            self.em_split = self.em.split_in_frames(self.frame_range[0], self.frame_range[1])

        if self.poolsize != 0:
            # with multiprocessing.Pool(processes=self.poolsize) as pool:
            #     # pool = ThreadPool(self.poolsize)
            #     frame_list = pool.starmap(self.forward_single_frame_wrapper, zip(self.em_split,
            #                                                                      iter.repeat(self.psf),
            #                                                                      iter.repeat(self.background)))
            raise NotImplementedError("Does not work at the moment.")

        elif self.poolsize == 0:
            em_sets = self.em_split.__len__()
            frame_list = [None] * em_sets
            for i in range(em_sets):
                frame_list[i] = self.forward_single_frame_wrapper(self.em_split[i],
                                                                  self.psf)

        frames = torch.stack(frame_list, dim=0)

        """Add background. This needs to happen here and not on a single frame, since background may be correlated."""
        if self.background is not None:
            self.frames = self.background.forward(frames).type(torch.int16)
        else:
            self.frames = frames.type(torch.int16)

        return self.frames

    def write_to_binary(self, outfile):
            """
            Writes frames and emitters to binary.
            :param outfile: output file
            :return: void
            """
            np.savez_compressed(outfile,
                                frames=self.frames.numpy(),
                                xyz=self.em.xyz,
                                phot=self.em.phot,
                                id=self.em.id,
                                frame_ix=self.em.frame_ix,
                                extent=self.extent)
            print("Saving simulation to {}.".format(outfile))


class SimulationArgs:
    def __init__(self, extent, img_shape, bg_value):
        self.extent = extent
        self.img_shape = img_shape
        self.bg_value = bg_value


if __name__ == '__main__':
    """Get root folder of this package."""
    deepsmlm_root = os.path.abspath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     os.pardir, os.pardir)) + '/'

    args = SimulationArgs(extent=((-0.5, 25.5), (-0.5, 25.5), (-200, 200)),
                          img_shape=(26, 26),
                          bg_value=None)
    args.csp_calib = deepsmlm_root + \
                     'data/Cubic Spline Coefficients/2019-02-19/000_3D_cal_640i_50_Z-stack_1_MMStack.ome_3dcal.mat'
    args.binary_path = deepsmlm_root + 'data/temp.npz'

    sp = SMAPSplineCoefficient(args.csp_calib)
    psf = sp.init_spline(args.extent[0], args.extent[1], img_shape=args.img_shape)
    bg = Poisson(bg_uniform=15)

    num_emitters = 100000
    em = EmitterSet(xyz=torch.rand((num_emitters, 3)) * torch.tensor([30., 30, 5]),
                    phot=torch.randint(800, 4000, (num_emitters,)),
                    frame_ix=torch.randint(0, 10000, (num_emitters,)))

    em = EmitterSet(xyz=torch.tensor([[15., 15, -200], [15., 15, -200]]),
                    phot=torch.tensor([2000, 1000.]),
                    frame_ix=torch.tensor([0., 0.]))

    sim = Simulation(em=em, extent=args.extent, psf=psf, background=bg, poolsize=0, frame_range=(0, 0))
    sim.forward(em)
    sim.write_to_binary(args.binary_path)
