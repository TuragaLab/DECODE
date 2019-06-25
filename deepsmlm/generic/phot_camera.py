import torch

import deepsmlm.generic.noise as noise


class Photon2Camera:
    def __init__(self, qe, bg_uniform, em_gain, e_per_adu, baseline, read_sigma):
        """

        :param qe: quantum efficiency
        :param bg_uniform: uniform background (before poisson)
        :param em_gain: em-gain
        :param e_per_adu: electrons per analog digital unit
        :param baseline: baseline (for non negative values)
        :param read_sigma: read-out sigma (in electrons)
        """
        self.qe = qe
        self.bg_uniform = bg_uniform
        self.em_gain = em_gain
        self.e_per_adu = e_per_adu
        self.baseline = baseline
        self.read_sigma = read_sigma

        self.poisson = noise.Poisson(bg_uniform=0)
        self.read = noise.Gaussian(sigma_gaussian=self.read_sigma,
                                   bg_uniform=0)

    def forward(self, x):
        """
        Input in photons
        :param x:
        :return:
        """
        camera = self.poisson.forward((x + self.bg_uniform) * self.qe)
        camera = self.read.forward(camera)
        camera *= self.em_gain / self.e_per_adu
        camera += self.baseline
        return camera
