import torch
import warnings

from . import noise


class Photon2Camera:
    def __init__(self, qe, spur_noise, em_gain, e_per_adu, baseline, read_sigma, photon_units,
                 device: (str, torch.device) = None):
        """

        Args:
            qe:
            spur_noise:
            em_gain:
            e_per_adu:
            baseline:
            read_sigma:
            photon_units:
            device: default device for forward and backward
        """
        self.qe = qe
        self.spur = spur_noise
        self.em_gain = em_gain
        self.e_per_adu = e_per_adu
        self.baseline = baseline
        self.read_sigma = read_sigma
        self.device = device

        self.poisson = noise.Poisson()
        self.gain = noise.Gamma(scale=self.em_gain)
        self.read = noise.Gaussian(sigma_gaussian=self.read_sigma)
        self.photon_units = photon_units

    def __str__(self):
        return f"Photon to Camera Converter.\n" + \
               f"Camera: QE {self.qe} | Spur noise {self.spur} | EM Gain {self.em_gain} | " + \
               f"e_per_adu {self.e_per_adu} | Baseline {self.baseline} | Readnoise {self.read_sigma}\n" + \
               f"Output in Photon units: {self.photon_units}"

    @classmethod
    def parse(cls, param, **kwargs):

        return Photon2Camera(qe=param.Camera.qe, spur_noise=param.Camera.spur_noise,
                             em_gain=param.Camera.em_gain, e_per_adu=param.Camera.e_per_adu,
                             baseline=param.Camera.baseline, read_sigma=param.Camera.read_sigma,
                             photon_units=param.Camera.convert2photons,
                             **kwargs)

    def forward(self, x: torch.Tensor, device: (str, torch.device) = None) -> torch.Tensor:
        """
        Forwards frame through camera

        Args:
            x (torch.Tensor): camera frame of dimension *, H, W
            device (str, torch.device): device for forward

        Returns:
            torch.Tensor
        """
        if device is not None:
            x = x.to(device)
        elif self.device is not None:
            x = x.to(self.device)

        """Clamp input to 0."""
        x = torch.clamp(x, 0.)
        """Poisson for photon characteristics of emitter (plus autofluorescence etc."""
        camera = self.poisson.forward(x * self.qe + self.spur)
        """Gamma for EM-Gain (EM-CCD cameras, not sCMOS)"""
        if self.em_gain is not None:
            camera = self.gain.forward(camera)
        """Gaussian for read-noise. Takes camera and adds zero centred gaussian noise."""
        camera = self.read.forward(camera)
        """Electrons per ADU, (floor function)"""
        camera /= self.e_per_adu
        camera = camera.floor()
        """Add Manufacturer baseline. Make sure it's not below 0."""
        camera += self.baseline
        camera = torch.max(camera, torch.tensor([0.]).to(camera.device))

        if self.photon_units:
            if self.em_gain is not None:
                camera = ((camera - self.baseline) / self.em_gain * self.e_per_adu - self.spur) / self.qe
            else:
                camera = ((camera - self.baseline) * self.e_per_adu - self.spur) / self.qe
        return camera

    def backward(self, x: torch.Tensor, device: (str, torch.device) = None) -> torch.Tensor:
        """
        Calculates the expected number of photons from a noisy image.

        Args:
            x:
            device:

        Returns:

        """

        if device is not None:
            x = x.to(device)
        elif self.device is not None:
            x = x.to(self.device)

        if self.photon_units:
            warnings.warn(UserWarning("You try to convert from ADU (camera units) back to photons although this camera simulator"
                          "already outputs photon units. Make sure you know what you are doing."))
        out = (x - self.baseline) * self.e_per_adu / self.em_gain
        out -= self.spur
        out /= self.qe
        return out
