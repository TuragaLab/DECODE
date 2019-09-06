import torch

import deepsmlm.generic.background
import deepsmlm.generic.noise
import deepsmlm.generic.noise as noise


class Photon2Camera:
    def __init__(self, qe, spur_noise, em_gain, e_per_adu, baseline, read_sigma):
        """

        :param qe: quantum efficiency
        :param spur_noise: spurious noise
        :param em_gain: em-gain
        :param e_per_adu: electrons per analog digital unit
        :param baseline: baseline (for non negative values)
        :param read_sigma: read-out sigma (in electrons)
        """
        self.qe = qe
        self.spur = spur_noise
        self.em_gain = em_gain
        self.e_per_adu = e_per_adu
        self.baseline = baseline
        self.read_sigma = read_sigma

        self.poisson = deepsmlm.generic.noise.Poisson()
        self.gain = noise.Gamma(scale=self.em_gain)
        self.read = noise.Gaussian(sigma_gaussian=self.read_sigma,
                                   bg_uniform=0)

    @staticmethod
    def parse(param: dict):
        """

        :param param: parameter dictonary
        :return:
        """
        return Photon2Camera(qe=param['Camera']['qe'], spur_noise=param['Camera']['spur_noise'],
                             em_gain=param['Camera']['em_gain'], e_per_adu=param['Camera']['e_per_adu'],
                             baseline=param['Camera']['baseline'], read_sigma=param['Camera']['read_sigma'])

    def forward(self, x):
        """
        Input in photons
        :param x:
        :return:
        """
        """Clamp input to 0."""
        x = torch.clamp(x, 0.)
        """Poisson for photon characteristics of emitter (plus autofluorescence etc."""
        camera = self.poisson.forward(x * self.qe + self.spur)
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

    def reverse(self, x):
        """
        Calculate (expected) number of photons
        :param x:
        :return:
        """
        out = (x - self.baseline) * self.e_per_adu / self.em_gain
        out -= self.spur
        out /= self.qe
        return out
