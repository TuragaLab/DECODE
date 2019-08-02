import torch

import deepsmlm.generic.noise as noise


class Photon2Camera:
    def __init__(self, qe, spur_noise, bg_uniform, em_gain, e_per_adu, baseline, read_sigma):
        """

        :param qe: quantum efficiency
        :param spur_noise: spurious noise
        :param bg_uniform: uniform background (before poisson)
        :param em_gain: em-gain
        :param e_per_adu: electrons per analog digital unit
        :param baseline: baseline (for non negative values)
        :param read_sigma: read-out sigma (in electrons)
        """
        self.qe = qe
        self.spur = spur_noise
        self.bg_uniform = bg_uniform
        self.em_gain = em_gain
        self.e_per_adu = e_per_adu
        self.baseline = baseline
        self.read_sigma = read_sigma

        self.poisson = noise.Poisson(bg_uniform=0)
        self.gain = noise.Gamma(scale=self.em_gain)
        self.read = noise.Gaussian(sigma_gaussian=self.read_sigma,
                                   bg_uniform=0)

    def forward(self, x):
        """
        Input in photons
        :param x:
        :return:
        """
        """Poisson for photon characteristics of emitter (plus autofluorescence etc."""
        camera = self.poisson.forward((x + self.bg_uniform) * self.qe + self.spur)
        """Gamma for EM-Gain (EM-CCD cameras, not sCMOS)"""
        if self.em_gain is not None:
            camera = self.gain.forward(camera)
        """Gaussian for read-noise. Takes camera and adds zero centred gaussian noise."""
        camera = self.read.forward(camera)
        """Electrons per ADU"""
        camera /= self.e_per_adu
        """Manufacturer baseline. Make sure it's not below 0."""
        camera += self.baseline
        camera = torch.max(camera, torch.tensor([0.]))
        return camera
