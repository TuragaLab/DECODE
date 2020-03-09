import math
from abc import ABC, abstractmethod  # abstract class
from collections import namedtuple
from functools import partial

import deprecated
import numpy as np
import torch
from scipy import interpolate

import deepsmlm.simulation.psf_kernel as psf_kernel


class Background(ABC):
    """
    Abstract background class. All inheritors must implement a forward method that takes a batch of frames and
    adds(!) the background to it.
    """

    bg_return = namedtuple('bg_return', ['xbg', 'bg'])  # return arguments, x plus bg and bg term alone

    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, x: torch.Tensor):
        """
        Takes a batch of frames and adds the implemented background component to it.

        Args:
            x (torch.Tensor): input frames. Dimension N x C x H x W

        Returns:
            xbg (torch.Tensor): x including background

        """
        bg = torch.zeros_like(x)
        return self.bg_return(xbg=x + bg, bg=bg)


class UniformBackground(Background):
    def __init__(self, bg_uniform: (float, tuple) = 0., bg_sampler=None):
        """
        Adds spatially constant background.

        Args:
            bg_uniform (float, tuple of floats): background value or background range. If tuple (bg range) the value
                will be sampled from a random uniform.
            bg_sampler (function): a custom bg sampler function
        """
        super().__init__()

        if (bg_uniform is not None) and (bg_sampler is not None):
            raise ValueError("You must either specify bg_uniform XOR a bg_distribution")

        if bg_sampler is None:
            if not (isinstance(bg_uniform, tuple) or isinstance(bg_uniform, list)):
                bg_uniform = [bg_uniform, bg_uniform]  # const. value

            self._bg_distribution = torch.distributions.uniform.Uniform(*bg_uniform).sample
        else:
            self._bg_distribution = bg_sampler

    @staticmethod
    def parse(param):
        return UniformBackground(param.Simulation.bg_uniform)

    def forward(self, x: torch.Tensor):
        """
        Takes a batch of frames and adds uniform background to it.

        Args:
            x (torch.Tensor): input frames. Dimension N x C x H x W

        Returns:
            xbg (torch.Tensor): x including background

        """
        bg_term = self._bg_distribution()
        return self.bg_return(xbg=x + bg_term, bg=bg_term)


class OutOfFocusEmitters(Background):
    """
    Simulate far out of focus emitters by using huge z values and a gaussian kernel.

    """

    def __init__(self, xextent: tuple, yextent: tuple, img_shape: tuple, ampl: tuple, num_oof_rg: tuple):
        """

        Args:
            xextent (tuple): extent in x
            yextent (tuple): extent in y
            img_shape (tuple of int): image shape
            ampl (tuple): amplitude of the background (peak value of the gaussian)
            num_oof_range (tuple): range of number of out-of-focus emitters
        """

        self.xextent = xextent
        self.yextent = yextent
        self.num_oof_rg = num_oof_rg

        self.gauss_psf = psf_kernel.GaussianExpect(xextent,
                                                   yextent,
                                                   (-5000., 5000.),
                                                   img_shape=img_shape,
                                                   sigma_0=2.5,
                                                   peak_weight=True)
        self.level_dist = torch.distributions.uniform.Uniform(low=ampl[0], high=ampl[1])
        self.num_emitter_dist = partial(torch.randint, low=self.num_oof_rg[0], high=self.num_oof_rg[1] + 1, size=(1,))

    @staticmethod
    def parse(param):
        return OutOfFocusEmitters(param.Simulation.psf_extent[0],
                                  param.Simulation.psf_extent[1],
                                  param.Simulation.img_size,
                                  param.Simulation.bg_oof_range,
                                  param.Simulation.bg_num_oof_range)

    def forward(self, x):
        """Sample emitters. Place them randomly over the image."""
        num_bg_em = self.num_emitter_dist().item()
        xyz = torch.rand((num_bg_em, 3))
        xyz *= torch.tensor([self.xextent[1] - self.xextent[0],
                             self.yextent[1] - self.yextent[0],
                             1.])
        xyz -= torch.tensor([self.xextent[0], self.yextent[0], 0.])
        xyz[:, 2] = torch.randint_like(xyz[:, 2], low=2000, high=8000)
        xyz[:, 2] *= torch.from_numpy(np.random.choice([-1., 1.], xyz.shape[0])).type(torch.FloatTensor)
        levels = self.level_dist.sample((xyz.size(0),))

        bg_term = self.gauss_psf.forward(xyz, levels)
        return self.bg_return(xbg=x + bg_term, bg=bg_term)


class PerlinBackground(Background):
    """
    Taken from https://gist.github.com/vadimkantorov/ac1b097753f217c5c11bc2ff396e0a57.
    """

    def __init__(self, img_size, perlin_scale: int, amplitude, draw_amp: bool = False):
        """

        :param img_size: size of the image
        :param perlin_scale: scale of the perlin in fraction of the img_scale
        :param amplitude: background strength
        :param draw_amp: draw the perlin amplitude from a uniform distribution
        """
        super().__init__()
        if img_size[0] != img_size[1]:
            raise ValueError("Currently only equal img-size supported.")

        self.img_size = img_size
        self.perlin_scale = perlin_scale
        self.amplitude = amplitude
        self.perlin_com = None
        self.draw_amp = draw_amp

        delta = (self.perlin_scale / self.img_size[0], self.perlin_scale / self.img_size[1])
        self.d = (self.img_size[0] // self.perlin_scale, self.img_size[1] // self.perlin_scale)
        self.grid = torch.stack(torch.meshgrid(torch.arange(0, self.perlin_scale, delta[0]),
                                               torch.arange(0, self.perlin_scale, delta[1])), dim=-1) % 1

    @staticmethod
    def parse(param):
        img_size = param.Simulation.img_size
        perlin_scale = param.Simulation.bg_perlin_scale
        amplitude = param.Simulation.bg_perlin_amplitude
        norm_amps = param.Simulation.bg_perlin_normalise_amplitudes
        draw_amps = param.Simulation.bg_perlin_draw_amps
        prob_disable = param.HyperParameter.bg_perlin_prob_disable

        if isinstance(amplitude, list) or isinstance(amplitude, tuple):
            return PerlinBackground.multi_scale_init(img_size=img_size,
                                                     scales=perlin_scale,
                                                     amps=amplitude,
                                                     norm_amps=norm_amps,
                                                     draw_amps=draw_amps,
                                                     prob_disable=prob_disable)
        else:
            return PerlinBackground(img_size=img_size,
                                    perlin_scale=perlin_scale,
                                    amplitude=amplitude,
                                    draw_amp=draw_amps)

    @staticmethod
    def multi_scale_init(**kwargs):
        """
        Generates a sequence of this class
        """
        return MultiPerlin(**kwargs)

    @staticmethod
    def fade_f(t):
        return 6 * t ** 5 - 15 * t ** 4 + 10 * t ** 3

    def calc_perlin(self, shape, res):

        if shape[0] == res[0] and shape[1] == res[1]:
            return torch.rand(*shape) * 2 - 1

        angles = 2 * math.pi * torch.rand(res[0] + 1, res[1] + 1)
        gradients = torch.stack((torch.cos(angles), torch.sin(angles)), dim=-1)

        tile_grads = lambda slice1, slice2: gradients[slice1[0]:slice1[1], slice2[0]:slice2[1]].repeat_interleave(
            self.d[0],
            0).repeat_interleave(
            self.d[1], 1)
        dot = lambda grad, shift: (
                torch.stack(
                    (self.grid[:shape[0], :shape[1], 0] + shift[0], self.grid[:shape[0], :shape[1], 1] + shift[1]),
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
        if self.draw_amp:
            amp_factor = torch.rand(1)
        else:
            amp_factor = 1.

        bg_term = self.amplitude * amp_factor * (self.calc_perlin(self.img_size, [self.perlin_scale,
                                                                                  self.perlin_scale]) + 1) / 2.0
        return self.bg_return(xbg=x + bg_term, bg=bg_term)


class MultiPerlin(Background):
    def __init__(self, img_size, scales, amps, draw_amps: bool, norm_amps: bool, prob_disable=None):
        """

        :param img_size: tuple of ints
        :param scales: perlin scales
        :param amps: amplitudes
        :param draw_amps: sample the amplitude in the respective scale range in every forward (from a uniform)
        :param norm_amps: normalise the amplitudes
        :param prob_disable: disable a frequency probabilistically
        """
        super().__init__()

        self.img_size = img_size
        self.scales = scales if not isinstance(scales, torch.Tensor) else torch.tensor(scales)
        self.amps = amps if isinstance(amps, torch.Tensor) else torch.tensor(amps)
        self.amps = self.amps.float()
        self.draw_amps = draw_amps
        self.num_freq = self.scales.__len__()  # number of frequencies
        self.norm_amps = norm_amps
        self.prob_disable = prob_disable

        if self.norm_amps:
            self.amps /= self.num_freq

        self.perlin_com = [PerlinBackground(self.img_size,
                                            self.scales[i],
                                            self.amps[i],
                                            draw_amp=self.draw_amps) for i in range(self.num_freq)]

    @staticmethod
    def parse(param):
        """
        Parse a param file. This dummy method calls the PerlinBackground Parser which is the unified version.
        :param param:
        :return:
        """
        return PerlinBackground.parse(param)

    def forward(self, x):
        """

        :param x:
        :return:
        """
        bg_term = torch.zeros((1, ))
        for i in range(self.num_freq):
            if (self.prob_disable is not None) and (torch.rand(1).item() <= self.prob_disable):
                continue
            x, bg_term_ = self.perlin_com[i].forward(x)  # temporary bg
            bg_term = bg_term + bg_term_  # book-keep bg_term

        """The behaviour here must be a bit different because the perlin components already add their bg_term to x."""
        return self.bg_return(xbg=x, bg=bg_term)


"""Deprecated Stuff."""
@deprecated.deprecated("Old implementation. Maybe for future investigation.")
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

        xn = np.linspace(0, 1, img_size[0])
        yn = np.linspace(0, 1, img_size[1])
        self.xn, self.yn = np.meshgrid(xn, yn)

    @staticmethod
    def parse(param):
        return NonUniformBackground(param.Simulation.bg_nonuni_intensity,
                                    param.Simulation.img_size,
                                    param.Simulation.bg_nonuni_dynamic)

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
