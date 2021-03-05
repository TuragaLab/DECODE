from abc import ABC, abstractmethod  # abstract class
import torch


class NoiseDistribution(ABC):
    """
    Abstract noise.
    """

    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Samples the noise distribution based on the input x.

        Args:
            x: input

        Returns:
            noisy sample
        """
        raise NotImplementedError


class ZeroNoise(NoiseDistribution):
    """
    The No-Noise noise.
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class Gamma(NoiseDistribution):
    """
    Gamma distribution.

    Attributes:
        scale: 1 / rate of the Gamma distribution

    """

    def __init__(self, scale: float):
        """

        Args:
            scale: 1 / rate of the Gamma distribution
        """
        super().__init__()
        self.scale = scale

    def forward(self, x):
        # disable validate_args because 0 is okay for sampling
        return torch.distributions.gamma.Gamma(x, 1 / self.scale, validate_args=False).sample()


class Gaussian(NoiseDistribution):
    """
    Gaussian distribution.

    Attributes:
        sigma (float, torch.Tensor): standard deviation fo the gaussian
    """

    def __init__(self, sigma: (float, torch.Tensor)):
        """

        Args:
            sigma: standard deviation fo the gaussian
        """
        super().__init__()
        self.sigma = sigma

    def forward(self, x):
        return x + self.sigma * torch.randn_like(x)


class Poisson(NoiseDistribution):
    """
    Poisson noise. 'Non-parametric' with respect  initialisation since the only parameter (lambda) comes from the input in the forward method itself.
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        # disable validate_args because 0 is okay for sampling
        return torch.distributions.poisson.Poisson(x, validate_args=False).sample()
