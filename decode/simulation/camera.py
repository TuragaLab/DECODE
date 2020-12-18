from abc import ABC, abstractmethod  # abstract class
from typing import Union

import torch
from deprecated import deprecated

from . import noise_distributions
from ..neuralfitter import sampling


class Camera(ABC):

    @abstractmethod
    def forward(self, x: torch.Tensor, device: Union[str, torch.device] = None) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def backward(self, x: torch.Tensor, device: Union[str, torch.device] = None) -> torch.Tensor:
        raise NotImplementedError


class Photon2Camera(Camera):
    """
    Simulates a physical EM-CCD camera device. Input are the theoretical photon counts as by the psf and background model,
    all the device specific things are modelled.

    """

    def __init__(self, *, qe: float, spur_noise: float, em_gain: Union[float, None], e_per_adu: float, baseline: float,
                 read_sigma: float, photon_units: bool, device: Union[str, torch.device] = None):
        """

        Args:
            qe: quantum efficiency :math:`0 ... 1'
            spur_noise: spurious noise
            em_gain: em gain
            e_per_adu: electrons per analog digital unit
            baseline: manufacturer baseline / offset
            read_sigma: readout sigma
            photon_units: convert back to photon units
            device: device (cpu / cuda)

        """
        self.qe = qe
        self.spur = spur_noise
        self._em_gain = em_gain
        self.e_per_adu = e_per_adu
        self.baseline = baseline
        self._read_sigma = read_sigma
        self.device = device

        self.poisson = noise_distributions.Poisson()
        self.gain = noise_distributions.Gamma(scale=self._em_gain)
        self.read = noise_distributions.Gaussian(sigma=self._read_sigma)
        self.photon_units = photon_units
        self.return_sigma_map = False

    @classmethod
    def parse(cls, param):

        return cls(qe=param.Camera.qe, spur_noise=param.Camera.spur_noise,
                   em_gain=param.Camera.em_gain, e_per_adu=param.Camera.e_per_adu,
                   baseline=param.Camera.baseline, read_sigma=param.Camera.read_sigma,
                   photon_units=param.Camera.convert2photons,
                   device=param.Hardware.device_simulation)

    def __str__(self):
        return f"Photon to Camera Converter.\n" + \
               f"Camera: QE {self.qe} | Spur noise {self.spur} | EM Gain {self._em_gain} | " + \
               f"e_per_adu {self.e_per_adu} | Baseline {self.baseline} | Readnoise {self._read_sigma}\n" + \
               f"Output in Photon units: {self.photon_units}"

    def forward(self, x: torch.Tensor, device: Union[str, torch.device] = None) -> torch.Tensor:
        """
        Forwards frame through camera

        Args:
            x: camera frame of dimension *, H, W
            device: device for forward

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
        if self._em_gain is not None:
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
            return self.backward(camera, device)

        return camera

    def backward(self, x: torch.Tensor, device: Union[str, torch.device] = None) -> torch.Tensor:
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

        out = (x - self.baseline) * self.e_per_adu
        if self._em_gain is not None:
            out /= self._em_gain
        out -= self.spur
        out /= self.qe

        return out


class PerfectCamera(Photon2Camera):
    def __init__(self, device: Union[str, torch.device] = None):
        """
        Convenience wrapper for perfect camera, i.e. only shot noise. By design in 'photon units'.

        Args:
            device: device for simulation

        """
        super().__init__(qe=1.0, spur_noise=0., em_gain=None, e_per_adu=1., baseline=0., read_sigma=0.,
                         photon_units=False, device=device)

    @classmethod
    def parse(cls, param):
        return cls(device=param.Hardware.device_simulation)


@deprecated(reason="Not yet ready implementation. Needs thorough testing and validation.")
class SCMOS(Photon2Camera):
    """
    Models a sCMOS camera. You can provide a pixel-wise sigma map of the readout noise of the camera.

    """

    def __init__(self, qe: float, spur_noise: float, em_gain: float, e_per_adu: float, baseline: float,
                 read_sigma: torch.Tensor, photon_units: bool, device: (str, torch.device) = None):

        super().__init__(qe=None, spur_noise=0., em_gain=em_gain, e_per_adu=1.,
                         baseline=baseline, read_sigma=read_sigma, photon_units=photon_units, device=device)

        self.return_sigma_map = True

#     def check_sanity(self):

#         if self._read_sigma.dim() != 2:
#             raise ValueError(f"Expected readout noise map to be 2D")

# #         if self.sample_mode not in ('batch', 'const'):
# #             raise ValueError(f"Sample mode: {self.sample_mode} not supported.")

#     def sample_sensor_window(self, size_nxy: tuple) -> torch.Tensor:
#         """
#         Samples a random window from the sensor and returns the corresponding readout noise values

#         Args:
#             size_nxy: number of samples and window size, i.e. tuple of len 3, where (N, H, W)

#         Returns:
#             read-out noise window samples

#         """
#         return sampling.sample_crop(self._read_sigma, size_nxy)

    def forward(self, x: torch.Tensor, device: Union[str, torch.device] = None) \
            -> (torch.Tensor, torch.Tensor):
        """
        Forwards model input image 'x' through camera.

        Args:
            x: model image
            device:

        Returns:
            Sampled noisy image
            Sampled noise map
        """

#         if torch.is_tensor(self._read_sigma):
#             if x.size() != self._read_sigma.size():

#                 if x.dim() == 4:
#                     sigma = self.sample_sensor_window((x.size(0), x.size(-2), x.size(-1)))
#                     sigma.unsqueeze_(1)
#                 else:
#                     sigma = self.sample_sensor_window((1, x.size(-2), x.size(-1)))
#                     sigma = sigma.unsqueeze_(0)

#                 self.read = noise_distributions.Gaussian(sigma.to(device))
#         else:

        """ Sample gamma distributed readout noise sigmas """

#         sigma = torch.distributions.Gamma(self._read_sigma[0], 1 / self._read_sigma[2]).sample(x.shape) + self._read_sigma[1]
        sigma = torch.distributions.Gamma(self._read_sigma[0], 1 / self._read_sigma[2]).sample(x[0].shape) + self._read_sigma[1]
        sigma = sigma[None].repeat(x.shape[0],1,1)
        self.read = noise_distributions.Gaussian(sigma.to(device))

        """Implement noise model as in https://www.nature.com/articles/nmeth.4379 (Supplement)"""

        """Clamp input to 0."""
        x = torch.clamp(x, 0.)

        """Poisson for photon characteristics of emitter"""
        camera = self.poisson.forward(x)

        """EM-Gain"""
        camera = camera * self._em_gain

        """Gaussian for read-noise. Takes camera and adds zero centred gaussian noise."""
        camera = self.read.forward(camera).to(camera.device)

        """Floor function"""
        camera = camera.floor()

        """Add Manufacturer baseline. Make sure it's not below 0."""
        camera += self.baseline
        camera = torch.max(camera, torch.tensor([0.]).to(camera.device))

        if self.photon_units:
            camera = self.backward(camera, device)

        return camera, sigma

    def backward(self, x: torch.Tensor, device: Union[str, torch.device] = None) -> torch.Tensor:
        """
        Substracts the baseline and scales by the gain. These can be scalars (for simulated data) or pixel maps for real data (if available)

        Args:
            x:
            device:

        Returns:

        """

        if device is not None:
            x = x.to(device)
        elif self.device is not None:
            x = x.to(self.device)

        if torch.is_tensor(self.baseline):
            assert self.baseline.ndim == 2 and x.shape[-2:] == self.baseline.shape, 'Baseline map needs to be of same size as the input frames'
        if torch.is_tensor(self._em_gain):
            assert self._em_gain.ndim == 2 and x.shape[-2:] == self._em_gain.shape, 'Gain map needs to be of same size as the input frames'

        out = (x - self.baseline) / self._em_gain

        return out
