import numpy as np
import matplotlib.pyplot as plt
import math
import numbers
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import signal, ndimage, special  # for gaussian convolution
from skimage.measure import block_reduce


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


def delta_psf(pos, p_count, img_shape=np.array([64, 64]), origin='px_centre', xextent=None, yextent=None):

    # as long as we don't have a good replacement in pytorch for histogram2d let this live in numpy
    pos = pos.numpy()
    p_count = p_count.numpy()

    shape = img_shape
    # extent of our coordinate system
    if xextent is None:
        xextent = np.array([0, shape[0]], dtype=float)
    if yextent is None:
        yextent = np.array([0, shape[1]], dtype=float)

    if origin == 'px_centre':  # shift 0 right towards the centre of the first px, and down
        xextent -= 0.5
        yextent -= 0.5
    '''
    Binning in numpy: binning is (left Bin, right Bin]
    (open left edge, including right edge)
    '''

    bin_rows = np.linspace(xextent[0], xextent[1], img_shape[0] + 1, endpoint=True)
    bin_cols = np.linspace(yextent[0], yextent[1], img_shape[1] + 1, endpoint=True)

    camera, _, _ = np.histogram2d(pos[:, 1], pos[:, 0], bins=(
        bin_rows, bin_cols), weights=p_count)  # bin into 2d histogram with px edges

    return torch.from_numpy(camera).unsqueeze(0)


def gaussian_expect(pos, sig, p_count=1000, img_shape=np.array([64, 64])):
    """
    Function to return gaussian psf based on errorfunction.
    You must not use this function without the noise line at the end.

    :param pos:
    :param sig:
    :param p_count:
    :param img_shape:
    :return:
    """

    num_emitters = pos.shape[0]
    if num_emitters == 0:
        return torch.zeros(1, img_shape[0], img_shape[1])

    # Old numpy code. Will be removed eventually
    # i = 0
    # xpos = np.repeat(pos[:, 0].reshape((1, 1, num_emitters)), img_shape[i], axis=i)
    # ypos = np.repeat(pos[:, 1].reshape((1, 1, num_emitters)), img_shape[i], axis=i)
    # i = 1
    # xpos = np.repeat(xpos, img_shape[i], axis=i)
    # ypos = np.repeat(ypos, img_shape[i], axis=i)

    # xx, yy = np.meshgrid(x, y)
    # xx = np.expand_dims(xx, 2)
    # yy = np.expand_dims(yy, 2)
    # xx = np.repeat(xx, num_emitters, axis=2)
    # yy = np.repeat(yy, num_emitters, axis=2)

    xpos = pos[:, 0].repeat(img_shape[0], img_shape[1], 1)
    ypos = pos[:, 1].repeat(img_shape[0], img_shape[1], 1)

    x = torch.arange(img_shape[0], dtype=torch.float32)
    y = torch.arange(img_shape[1], dtype=torch.float32)
    xx, yy = torch.meshgrid(x, y)

    xx = xx.transpose(0, 1).unsqueeze(2).repeat(1, 1, num_emitters)
    yy = yy.transpose(0, 1).unsqueeze(2).repeat(1, 1, num_emitters)

    gauss_x = torch.erf((xx - xpos + 0.5) / (math.sqrt(2) * sig[0])) \
            - torch.erf((xx - xpos - 0.5) / (math.sqrt(2) * sig[0]))
    gauss_y = torch.erf((yy - ypos + 0.5) / (math.sqrt(2) * sig[1])) \
            - torch.erf((yy - ypos - 0.5) / (math.sqrt(2) * sig[1]))
    gaussCdf = p_count / 4 * torch.mul(gauss_x, gauss_y)

    gaussCdf = torch.sum(gaussCdf, 2)
    return torch.distributions.poisson.Poisson(gaussCdf).sample().unsqueeze(0)


def gaussian_convolution(pos,
                    sig=(2, 2), photon_count=1000, img_shape=np.array([64, 64]), up_fact=10):
    """
    Function to compute the gaussian PSF with a convolution. Grid based.

    :param pos:
    :param sig:
    :param photon_count:
    :param img_shape:
    :param up_fact:

    :return:
    """

    # use a finer grid, transform coordinates from px to fine-px grid
    _pos = (0.5 + pos) * up_fact # not yet understood why it's off by half a pixel
    _sig = up_fact * sig
    img_shape_hr = np.array(img_shape) * up_fact

    # compute kernel. is this the cdf?
    kernel = np.outer(signal.gaussian(np.ceil(10 * _sig[0]), _sig[0]),
                      signal.gaussian(np.ceil(10 * _sig[1]), _sig[1]))
    kernel = kernel / np.sum(kernel)

    img_hr = convolutional_sample(_pos, photon_count, kernel, img_shape_hr)

    # downsample
    img_lr = block_reduce(img_hr, block_size=(up_fact, up_fact), func=np.sum)

    raise NotImplementedError('Image vector has wrong dimensionality. Need to add singleton dimension for channels.')

    return img_lr


def convolutional_sample(pos, p_count, kernel, img_shape):  # this convolution could be done with pytorch on the gpu
    """
    Function to convolute an image made up ouf single positions with a kernel

    :param pos:
    :param p_count:
    :param kernel:
    :param img_shape:
    :return:
    """
    # round positions to integer
    pos = np.round(pos).astype(int)
    pos = np.clip(pos, 0, img_shape[0] - 1)

    img = np.zeros((img_shape[0], img_shape[1]))
    img[pos[:, 1], pos[:, 0]] = p_count  # this effectively is the centre of the px
    img = np.clip(signal.fftconvolve(img, kernel, mode='same'), 0, None)
    img = np.random.poisson(img)  # inherent poisson noise

    raise NotImplementedError('Image vector has wrong dimensionality. Need to add singleton dimension for channels.')

    return img


def astigmatism(pos):
    """
    Dummy function to test the ability of astigmatism

    :param pos:

    :return:
    """
    if pos.__len__() == 3:  # when we are in 3D
        if pos > 0:
            cov[0, 0] = sig**2 + 0.2 * pos
        elif pos < 0:
            cov[-1, -1] = sig**2 - 0.2 * pos

    sigma_x = np.sqrt(cov[0, 0])
    sigma_y = np.sqrt(cov[-1, -1])
    return cov, sigma_x, sigma_y


def gaussian_binned_sampling(pos, sig=(2, 2), photon_count=100, img_shape=(64, 64), origin='px_centre', xextent=None, yextent=None):
    """
    Most natural gaussian psf function. Though, it's slow but the gold standard.

    :param pos:
    :param sig:
    :param photon_count:
    :param img_shape:
    :param origin:
    :param xextent:
    :param yextent:

    :return:
    """
    mu = pos[:2]
    cov = np.power(np.array([[sig[0], 0], [0, sig[1]]]), 2)
    cov, _, _ = astigmatism(pos, cov)
    phot_pos = np.random.multivariate_normal(mu, cov, int(photon_count))  # in nm

    shape = img_shape
    # extent of our coordinate system
    if xextent is None:
        xextent = np.array([0, shape[0]], dtype=float)
    if yextent is None:
        yextent = np.array([0, shape[1]], dtype=float)

    if origin == 'px_centre':  # shift 0 right towards the centre of the first px, and down
        xextent -= 0.5
        yextent -= 0.5
    '''
    Binning in numpy: binning is (left Bin, right Bin]
    (open left edge, including right edge)
    '''
    bin_rows = np.linspace(xextent[0], xextent[1], img_shape[0] + 1, endpoint=True)
    bin_cols = np.linspace(yextent[0], yextent[1], img_shape[1] + 1, endpoint=True)

    camera, xedges, yedges = np.histogram2d(phot_pos[:, 1], phot_pos[:, 0], bins=(
        bin_rows, bin_cols))  # bin into 2d histogram with px edges

    raise NotImplementedError('Image vector has wrong dimensionality. Need to add singleton dimension for channels.')

    return camera


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


def weighted_avg_std(values, weights):
    """
    https://stackoverflow.com/questions/2413522/weighted-standard-deviation-in-numpy
    Return the weighted average and standard deviation.
    values, weights -- Numpy ndarrays with the same shape.
    """
    average = np.average(values, weights=weights)
    variance = np.average((values-average)**2, weights=weights)
    return average, np.sqrt(variance)


if __name__ == '__main__':
    im_size = 20
    im_extent = [-0.5, im_size-0.5, im_size-0.5, -0.5]

    img1 = np.zeros((im_size, im_size))
    img2 = np.zeros((40, 40))
    positions = np.array(([[9, 5], [12, 14]]))
    sigma = np.array([1.5, 1.5])
    cov = np.array(([[4, 0], [0, 4]]))
    photons = np.array([1000, 500])

    # img1 += gaussian_binned_sampling(img1, pos[1, :], photons)
    # img2 = gaussian_convolution(positions, sigma, photons, img2.shape)
    img1 = gaussian_expect(positions, sigma, photons, img1.shape)
    img2 = delta_psf(positions, photons, img2.shape, im_extent,
                     xextent=np.array([0, img1.shape[0]]), yextent=np.array([0, img1.shape[1]]))

    plt.figure()
    plt.subplot(221)
    plt.imshow(img1, cmap='gray', extent=im_extent)

    plt.subplot(223)
    plt.imshow(img2, cmap='gray', extent=im_extent)
    #
    # plt.subplot(222)
    # plt.bar(range(im_size), np.sum(img1, axis=0))
    # #
    # plt.subplot(224)
    # plt.bar(range(im_size), np.sum(img2, axis=0))




    plt.show()
    # print(weighted_avg_std(range(im_size), np.sum(img1, axis=0)))
    # print(weighted_avg_std(range(im_size), np.sum(img1, axis=1)))
    # print(weighted_avg_std(range(im_size), np.sum(img2, axis=0)))
    # print(weighted_avg_std(range(im_size), np.sum(img2, axis=1)))
    print("Done.")
