import torch

import deepsmlm.generic.emitter as emitter
import deepsmlm.generic.psf_kernel as psf_kernel


class ExperimentBg:
    """A class where we have a background which is constant over the course of the whole experiment, and """
    def __init__(self, xextent, yextent):
        self.psf = psf_kernel.GaussianExpect(xextent, yextent, (750., 5000.))

    def forward(self, x):
        self.psf.forward(x)
