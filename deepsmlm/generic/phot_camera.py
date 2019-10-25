import torch
import warnings

import deepsmlm.generic.background
import deepsmlm.generic.noise
import deepsmlm.generic.noise as noise


class Photon2Camera:
    def __init__(self, qe, spur_noise, em_gain, e_per_adu, baseline, read_sigma, photon_units):
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
        self.photon_units = photon_units

    def __str__(self):
        return f"Photon to Camera Converter.\n" + \
               f"Camera: QE {self.qe} | Spur noise {self.spur} | EM Gain {self.em_gain} | " + \
               f"e_per_adu {self.e_per_adu} | Baseline {self.baseline} | Readnoise {self.read_sigma}" + \
               f"Output in Photon units: {self.photon_units}"

    @staticmethod
    def parse(param: dict):
        """

        :param param: parameter dictonary
        :return:
        """
        return Photon2Camera(qe=param['Camera']['qe'], spur_noise=param['Camera']['spur_noise'],
                             em_gain=param['Camera']['em_gain'], e_per_adu=param['Camera']['e_per_adu'],
                             baseline=param['Camera']['baseline'], read_sigma=param['Camera']['read_sigma'],
                             photon_units=param['Camera'][''])

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
        """Add Manufacturer baseline and round the values since the camera will out int.  Make sure it's not below 0."""
        camera += self.baseline
        camera = camera.round()
        camera = torch.max(camera, torch.tensor([0.]))

        if self.photon_units:
            if self.em_gain is not None:
                camera = (camera - self.baseline) / self.em_gain * self.e_per_adu
            else:
                camera = camera - self.baseline * self.e_per_adu
        return camera

    def reverse(self, x):
        """
        Calculate (expected) number of photons
        :param x:
        :return:
        """
        if self.photon_units:
            warnings.warn(UserWarning("You try to convert from ADU (camera units) back to photons although this camera simulator"
                          "already outputs photon units. Make sure you know what you are doing."))
        out = (x - self.baseline) * self.e_per_adu / self.em_gain
        out -= self.spur
        out /= self.qe
        return out
