import abc  # abstract class
import torch


class Noise_Post(ABC):

    @abstractmethod
    def __init__(self):
        super.__init__(self)

    @abstractmethod
    def forward(self, image=None):
        pass


def noise_psf(img, noise=True, bg_poisson=10, readout_gaussian=0):
    """
    Function to add noise to an image

    :param img: image array
    :param noise:  add noise?
    :param bg_poisson:  constant background poissonian
    :param readout_gaussian:  readout gaussian, needs double check

    :return: img with noise
    """
    if not noise:
        return img

    noise_mask_poisson = torch.distributions.poisson.Poisson(torch.zeros_like(img) + bg_poisson).sample()  # not additive
    noise_mask_gaussian = readout_gaussian * torch.randn(img.shape[0], img.shape[1])  # additive
    return img + noise_mask_poisson + noise_mask_gaussian


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

    def __init__(self, channels, kernel_size, sigma, dim=2, cuda=False, padding=lambda x: x):
        super().__init__()
        if isinstance(kernel_size, numbers.Number):
            kernel_size = [kernel_size] * dim
        if isinstance(sigma, numbers.Number):
            sigma = [sigma] * dim

        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid(
            [
                torch.arange(size, dtype=torch.float32)
                for size in kernel_size
            ]
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= 1 / (std * math.sqrt(2 * math.pi)) * \
                      torch.exp(-((mgrid - mean) / (2 * std)) ** 2)

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
