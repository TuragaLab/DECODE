import numpy as np
import torch

import deepsmlm.generic.emitter as emitter
import deepsmlm.generic.psf_kernel as psf_kernel


class OutOfFocusEmitters:
    """Simulate out of focus emitters by using huge z values."""
    def __init__(self, xextent, yextent, img_shape, bg_range=(15, 15), num_bg_emitter=0):
        """

        :param xextent:
        :param yextent:
        :param img_shape:
        :param bg_range: peak height of emitters
        :param num_bg_emitter: number of background emitters in image.
        """

        self.xextent = xextent
        self.yextent = yextent
        self.num_bg_emitter = num_bg_emitter
        self.psf = psf_kernel.GaussianExpect(xextent,
                                             yextent,
                                             (-5000., 5000.),
                                             img_shape=img_shape,
                                             sigma_0=1.5,
                                             peak_weight=True)
        self.level_dist = torch.distributions.uniform.Uniform(low=bg_range[0], high=bg_range[1])

    def forward(self, x):
        """
        Inputs a batch of frames and adds bg to all of them
        :param x: NCHW
        :return:
        """
        """Sample emitters. Place them randomly over the image."""
        xyz = torch.rand((self.num_bg_emitter, 3)) * torch.tensor([self.xextent[1] - self.xextent[0],
                                                 self.yextent[1] - self.yextent[0],
                                                 1.]) - torch.tensor([self.xextent[0], self.yextent[0], 0.])
        xyz[:, 2] = torch.randint_like(xyz[:, 2], low=2000, high=8000)
        xyz[:, 2] *= torch.from_numpy(np.random.choice([-1., 1.], xyz.shape[0])).type(torch.FloatTensor)
        levels = self.level_dist.sample((xyz.size(0),))

        return x + self.psf.forward(xyz, levels)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import deepsmlm.generic.plotting.frame_coord as smplot

    extent = ((-0.5, 63.5), (-0.5, 63.5), (-750., 750.))
    img_shape = (64, 64)
    bg = OutOfFocusEmitters(extent[0], extent[1], img_shape, bg_range=(10., 20.), num_bg_emitter=3)

    x = torch.zeros((1, 1, 64, 64))
    out = bg.forward(x)
    plt.imshow(out[0, 0]); plt.colorbar(); plt.show()

