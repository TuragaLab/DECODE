from abc import ABC, abstractmethod  # abstract class
import numbers
import math
import torch
import torch.nn.functional as F
import torch.nn as nn


class NoisePost(ABC):
    """
    Abstract class of noise functions.
    """

    @abstractmethod
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, image):
        """
        Evaluate noise on the image.

        :param image: torch.tensor of arbitrary dimensionality.
        :return: image with noise.
        """
        return None


class IdentityNoise(NoisePost):
    """Dummy class which does not do anything."""

    def __init__(self):
        super().__init__()

    def forward(self, image):
        """Return unmodified image."""
        return image


class Poisson(NoisePost):

    def __init__(self, bg_uniform=0):
        """

        :param bg_uniform: uniform background value to be added before drawing from poisson distribution
        """
        super().__init__()

        self.bg_uniform = bg_uniform

    def forward(self, image):
        return torch.distributions.poisson.Poisson(image + self.bg_uniform).sample()


class Gaussian(NoisePost):

    def __init__(self, sigma_gaussian, bg_uniform):
        """

        :param sigma_gaussian: sigma value of gauss noise
        :param bg_uniform: uniform value to be added
        """
        super().__init__()

        self.sigma_gaussian = sigma_gaussian
        self.bg_uniform = bg_uniform

    def forward(self, image):

        return image + self.bg_uniform + self.sigma_gaussian * torch.randn(image.shape[0], image.shape[1])


class GaussianSmoothing(nn.Module):
    """
    Based on: https://discuss.pytorch.org/t/is-there-anyway-to-do-gaussian-filtering-for-an-image-2d-3d-in-pytorch/12351/8

    Apply gaussian smoothing on a
    1d, 2d or 3d tensor. Filtering is performed seperately for each channel
    in the input using a depthwise convolution.
    Arguments:
        channels (int, sequence): Number of channels of the input tensors. Output will
            have this number of channels as well.
        kernel_size (int, sequence): Size of the gaussian kernel.
        sigma (float, sequence): Standard deviation of the gaussian kernel.
        dim (int, optional): The number of dimensions of the data.
            Default value is 2 (spatial).

    Example usage:
    smoothing = GaussianSmoothing(3, 5, 1)
    input = torch.rand(1, 3, 100, 100)
    input = F.pad(input, (2, 2, 2, 2), mode='reflect')
    output = smoothing(input)
    """

    def __init__(self, channels, kernel_size, sigma, dim=2, cuda=False, padding=lambda x: x, kernel_f='gaussian'):
        super().__init__()
        if isinstance(kernel_size, numbers.Number):
            kernel_size = [kernel_size] * dim
        if isinstance(sigma, numbers.Number):
            sigma = [sigma] * dim
        self.kernel_f = kernel_f

        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid(
            [
                torch.arange(size, dtype=torch.float32)
                for size in kernel_size
            ]
        )
        if self.kernel_f == 'gaussian':
            for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
                mean = (size - 1) / 2
                kernel *= 1 / (std * math.sqrt(2 * math.pi)) * \
                          torch.exp(-((mgrid - mean) / (2 * std)) ** 2)
        elif self.kernel_f == 'laplacian':
            for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
                mean = (size - 1) / 2
                kernel *= 1 / (2 * std) * torch.exp(-torch.abs((mgrid - mean)) / std)
        else:
            raise ValueError("Mode must either be gaussian or laplacian.")

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)

        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        if cuda:
            kernel = kernel.cuda()

        self.register_buffer('weight', kernel)
        self.groups = channels

        if dim == 1:
            self.conv = F.conv1d
        elif dim == 2:
            self.conv = F.conv2d
        elif dim == 3:
            self.conv = F.conv3d
        else:
            raise RuntimeError(
                'Only 1, 2 and 3 dimensions are supported. Received {}.'.format(dim)
            )

        self.padding = padding  # padding function

    def forward(self, input, padding=None):
        """
        Apply gaussian filter to input.
        Arguments:
            input (torch.Tensor): Input to apply gaussian filter on.
        Returns:
            filtered (torch.Tensor): Filtered output.
        """

        return self.conv(self.padding(input), weight=self.weight, groups=self.groups)


if __name__ == '__main__':
    smoothener = GaussianSmoothing