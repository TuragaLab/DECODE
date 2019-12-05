from abc import ABC, abstractmethod  # abstract class

import numpy as np
import torch
from scipy import interpolate
import math

import deepsmlm.generic.psf_kernel as psf_kernel
import deepsmlm.generic.utils.processing as proc


class Background(ABC):

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def forward(self, x):
        """
        Must implement this forward method with frame input N C H W
        :param x:
        :return:
        """
        return


class UniformBackground(Background):
    def __init__(self, bg_uniform=0., bg_sampler=None):
        """

        :param bg_uniform: float or list / tuple. if the latter, bg value will be from a distribution
        :param bg_sampler: provide any function which upon call returns a value
        """
        super().__init__()

        if (bg_uniform is not None) and (bg_sampler is not None):
            raise ValueError("You must either specify bg_uniform (X)OR a bg_distribution")

        if bg_sampler is None:
            if not (isinstance(bg_uniform, tuple) or isinstance(bg_uniform, list)):
                bg_uniform = [bg_uniform, bg_uniform]

            self.bg_distribution = torch.distributions.uniform.Uniform(*bg_uniform).sample
        else:
            self.bg_distribution = bg_sampler

    @staticmethod
    def parse(param):
        return UniformBackground(param['Simulation']['bg_uniform'])

    def forward(self, x):
        """
        Add uniform background
        :param x:
        :return:
        """
        return x + self.bg_distribution()


class NonUniformBackground(Background):
    """
    A class to produce nonuniform background which is done by placing 5 points with 5 different values
    on somewhat random positions and then interpolate an image.
    """
    def __init__(self, intensity, img_size, dynamic_factor=1.3):
        super().__init__()
        self.max_value = intensity
        self.dynamic_factor = dynamic_factor

        self.x = np.array([-0.3, -0.3, 0.5, 1.3, 1.3])
        self.y = np.array([-0.3, 1.3, 0.5, -0.3, 1.3])
        # self.x = np.array([-1., -1., 2., 2.])
        # self.y = np.array([-1., 2., -1., 2.])

        xn = np.linspace(0, 1, img_size[0])
        yn = np.linspace(0, 1, img_size[1])
        self.xn, self.yn = np.meshgrid(xn, yn)

    @staticmethod
    def parse(param):
        return NonUniformBackground(param['Simulation']['bg_nonuni_intensity'],
                                    param['Simulation']['img_size'],
                                    param['Simulation']['bg_nonuni_dynamic'])

    def forward(self, input):
        """
        :param x:
        :return:
        """
        """Simulate locs and values"""
        x = self.x + np.random.randn(self.x.shape[0]) * .2
        y = self.y + np.random.randn(self.y.shape[0]) * .2
        v = np.random.rand(x.shape[0]) * self.max_value / self.dynamic_factor
        v[2] *= self.dynamic_factor
        # Setup interpolation function
        f = interpolate.Rbf(x, y, v, function='gaussian')
        # Interpolate, convert and clamp to 0
        bg = f(self.xn, self.yn)
        bg = torch.clamp(torch.from_numpy(bg.astype('float32')), 0.)
        return input + bg.unsqueeze(0).unsqueeze(0).repeat(1, input.size(1), 1, 1)


class OutOfFocusEmitters:
    """Simulate out of focus emitters by using huge z values."""
    def __init__(self, xextent, yextent, img_shape, bg_range=(15, 15), num_bgem_range=0):
        """

        :param xextent:
        :param yextent:
        :param img_shape:
        :param bg_range: peak height of emitters
        :param num_bgem_range: number of background emitters in image.
        """

        self.xextent = xextent
        self.yextent = yextent
        self.num_bg_emitter = num_bgem_range
        self.psf = psf_kernel.GaussianExpect(xextent,
                                             yextent,
                                             (-5000., 5000.),
                                             img_shape=img_shape,
                                             sigma_0=2.5,
                                             peak_weight=True)
        self.level_dist = torch.distributions.uniform.Uniform(low=bg_range[0], high=bg_range[1])
        self.num_emitter_dist = torch.distributions.uniform.Uniform(low=num_bgem_range[0], high=num_bgem_range[1])

    @staticmethod
    def parse(param):
        return OutOfFocusEmitters(param['Simulation']['psf_extent'][0],
                                  param['Simulation']['psf_extent'][1],
                                  param['Simulation']['img_size'],
                                  param['Simulation']['bg_oof_range'],
                                  param['Simulation']['bg_num_oof_range'])

    def forward(self, x):
        """
        Inputs a batch of frames and adds bg to all of them
        :param x: NCHW
        :return:
        """
        """Sample emitters. Place them randomly over the image."""
        num_bg_em = self.num_emitter_dist.sample((1,)).int().item()
        xyz = torch.rand((num_bg_em, 3)) * torch.tensor([self.xextent[1] - self.xextent[0],
                                                 self.yextent[1] - self.yextent[0],
                                                 1.]) - torch.tensor([self.xextent[0], self.yextent[0], 0.])
        xyz[:, 2] = torch.randint_like(xyz[:, 2], low=2000, high=8000)
        xyz[:, 2] *= torch.from_numpy(np.random.choice([-1., 1.], xyz.shape[0])).type(torch.FloatTensor)
        levels = self.level_dist.sample((xyz.size(0),))

        return x + self.psf.forward(xyz, levels)


class PerlinBackground(Background):
    """
    Taken from https://gist.github.com/vadimkantorov/ac1b097753f217c5c11bc2ff396e0a57.
    """

    def __init__(self, img_size, perlin_scale: int, amplitude, prob_disable=None):
        """

        :param img_size: size of the image
        :param perlin_scale: scale of the perlin in fraction of the img_scale
        :param amplitude: background strength
        :param prob_disable: disable perlin background in prob_disable fraction of calls.
        """
        super().__init__()
        if img_size[0] != img_size[1]:
            raise ValueError("Currently only equal img-size supported.")

        self.img_size = img_size
        self.perlin_scale = perlin_scale
        self.amplitude = amplitude
        self.perlin_com = None
        self.prob_disable = prob_disable

        """
        If perlin_scale is a list of lists, and amplitude a list we can use multiple instances of this class to build up multiple scales (octaves). The instances are then in perlin_com (ponents).
        """
        if isinstance(amplitude, list) or isinstance(amplitude, tuple):
            pass
        else:
            num_instances = 1

        delta = (self.perlin_scale / self.img_size[0], self.perlin_scale / self.img_size[1])
        self.d = (self.img_size[0] // self.perlin_scale, self.img_size[1] // self.perlin_scale)
        self.grid = torch.stack(torch.meshgrid(torch.arange(0, self.perlin_scale, delta[0]),
                                               torch.arange(0, self.perlin_scale, delta[1])), dim=-1) % 1

    @staticmethod
    def parse(param):
        img_size = param['Simulation']['img_size']
        perlin_scale = param['Simulation']['bg_perlin_scale']
        amplitude = param['Simulation']['bg_perlin_amplitude']
        norm_amps = param['Simulation']['bg_perlin_normalise_amplitudes']
        prob_disable = param['HyperParameter']['bg_perlin_prob_disable']

        if isinstance(amplitude, list) or isinstance(amplitude, tuple):
            return PerlinBackground.multi_scale_init(img_size=img_size,
                                                     perlin_scales=perlin_scale,
                                                     amplitudes=amplitude,
                                                     norm_amplitudes=norm_amps,
                                                     prob_disable=prob_disable)
        else:
            return PerlinBackground(img_size=param['Simulation']['img_size'],
                                    perlin_scale=param['Simulation']['bg_perlin_scale'],
                                    amplitude=param['Simulation']['bg_perlin_amplitude'],
                                    prob_disable=prob_disable)

    @staticmethod
    def multi_scale_init(img_size, perlin_scales, amplitudes, norm_amplitudes=True, prob_disable=None):
        """
        Generates a sequence of this class
        """
        num_instances = amplitudes.__len__()
        com = [None] * num_instances
        if norm_amplitudes:
            normfactor = 1. / num_instances
        else:
            normfactor = 1.

        for i in range(num_instances):
            com[i] = PerlinBackground(img_size, perlin_scales[i], amplitudes[i] * normfactor, prob_disable)

        return proc.TransformSequence(com)

    def cast_sequence(self):
        self.__class__ = proc.TransformSequence

    @staticmethod
    def fade_f(t):
        return 6 * t ** 5 - 15 * t ** 4 + 10 * t ** 3

    def calc_perlin(self, shape, res):

        if shape[0] == res[0] and shape[1] == res[1]:
            return torch.rand(*shape) * 2 - 1

        angles = 2 * math.pi * torch.rand(res[0] + 1, res[1] + 1)
        gradients = torch.stack((torch.cos(angles), torch.sin(angles)), dim=-1)

        tile_grads = lambda slice1, slice2: gradients[slice1[0]:slice1[1], slice2[0]:slice2[1]].repeat_interleave(self.d[0],
                                                                                                                  0).repeat_interleave(
            self.d[1], 1)
        dot = lambda grad, shift: (
                torch.stack((self.grid[:shape[0], :shape[1], 0] + shift[0], self.grid[:shape[0], :shape[1], 1] + shift[1]),
                            dim=-1) * grad[:shape[0], :shape[1]]).sum(dim=-1)

        n00 = dot(tile_grads([0, -1], [0, -1]), [0, 0])
        n10 = dot(tile_grads([1, None], [0, -1]), [-1, 0])
        n01 = dot(tile_grads([0, -1], [1, None]), [0, -1])
        n11 = dot(tile_grads([1, None], [1, None]), [-1, -1])
        t = self.fade_f(self.grid[:shape[0], :shape[1]])
        return math.sqrt(2) * torch.lerp(torch.lerp(n00, n10, t[..., 0]), torch.lerp(n01, n11, t[..., 0]), t[..., 1])

    def forward(self, x):
        """
        Forwards the bg.
        :param x:
        :return:
        """
        """
        Probabilistically disable perlin background. VW Abgastest style
        Note: In MultiScale Perlin, this only disables one component / scale. The likelihood that all / none are
        on / off is therefore (1-p)^num_scales, or p^(num_scales)
        """
        if self.prob_disable is not None:
            if torch.rand(1).item() <= self.prob_disable:
                return x

        return x + self.amplitude * (self.calc_perlin(self.img_size, [self.perlin_scale, self.perlin_scale]) + 1) / 2.0


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    extent = ((-0.5, 31.5), (-0.5, 31.5), (-750., 750.))
    img_shape = (32, 32)
    # bg = OutOfFocusEmitters(extent[0], extent[1], img_shape, bg_range=(0., 100.), num_bgem_range=[0, 5])
    bg = PerlinBackground.multi_scale_init(img_shape, [32, 16, 8, 4], [1., 1., 1., 1.])
    x = torch.zeros((1, 1, 32, 32))
    x = bg.forward(x)

    plt.imshow(x[0, 0], interpolation='lanczos')
    plt.colorbar()
    plt.show()

    plt.imshow(x[0, 0])
    plt.colorbar()
    plt.show()
    #
    # bg2 = NonUniformBackground(100, img_shape, 1.0)
    # # x = torch.zeros((1, 1, 64, 64))
    # x = bg2.forward(x)
    # plt.imshow(x[0, 0]);
    # plt.colorbar()
    # plt.show()




