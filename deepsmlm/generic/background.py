from abc import ABC, abstractmethod  # abstract class
import numpy as np
import torch
import scipy
from scipy import interpolate

import deepsmlm.generic.psf_kernel as psf_kernel


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
        f = interpolate.Rbf(x, y, v, function='gaussian')
        bg = torch.clamp(torch.from_numpy(f(self.xn, self.yn).astype('float32')), 0.)
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


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    extent = ((-0.5, 31.5), (-0.5, 31.5), (-750., 750.))
    img_shape = (32, 32)
    bg = OutOfFocusEmitters(extent[0], extent[1], img_shape, bg_range=(0., 100.), num_bgem_range=[0, 5])

    x = torch.zeros((1, 1, 32, 32))
    x = bg.forward(x)
    plt.imshow(x[0, 0]); plt.colorbar(); plt.show()

    bg2 = NonUniformBackground(100, img_shape, 1.0)
    # x = torch.zeros((1, 1, 64, 64))
    x = bg2.forward(x)
    plt.imshow(x[0, 0]);
    plt.colorbar()
    plt.show()
