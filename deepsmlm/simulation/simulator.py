import itertools as iter
import math
import numpy as np
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

        else:
            em_sets = self.em_split.__len__()
            frame_list = [None] * em_sets
            for i in range(em_sets):
                frame_list[i] = self.forward_single_frame_wrapper(self.em_split[i],
                                                                  self.psf)

        frames = torch.stack(frame_list, dim=0)

        """Add background. This needs to happen here and not on a single frame, since background may be correlated."""
        if self.background is not None:
            self.frames = self.background.forward(frames).type(torch.int64)
        else:
            self.frames = frames.type(torch.int64)

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

    # load spline calibration
    calib_file = deepsmlm_root + 'data/Calibration/SMLM Challenge Beads/Coefficients Big ROI/AS-Exp_100nm_3dcal.mat'
    psf = SMAPSplineCoefficient(calib_file).init_spline((-0.5, 63.5), (-0.5, 63.5), (64, 64))

    noise = Photon2Camera(0.9, 0., 90., 300., 100., 10., 0.)

    simulator = Simulation(None, ((-0.5, 63.5), (-0.5, 63.5), None), psf, noise, poolsize=0, frame_range=(0, 0))

    em = RandomEmitterSet(5, 64)
    em.phot *= 1000000

    img = simulator.forward(em_new=em)
    PlotFrame(img[0]).plot()
    plt.show()