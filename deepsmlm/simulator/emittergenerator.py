from abc import ABC, abstractmethod  # abstract class
import math
import numpy as np
import random
from random import randint
import torch
from torch.distributions.exponential import Exponential

from deepsmlm.generic.emitter import EmitterSet, LooseEmitterSet


class EmitterGenerator(ABC):

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def generate_set(self):
        pass

    @abstractmethod
    def pop_single_frame(self):
        pass


class EmitterPopper:

    def __init__(self, xextent, yextent, zextent, density, photon_range, emitter_av=None):
        super().__init__()
        self.xextent = xextent
        self.yextent = yextent
        self.zextent = zextent
        self.density = density
        self.photon_range = photon_range

        self.scale = torch.tensor([(self.xextent[1] - self.xextent[0]),
                                   (self.yextent[1] - self.yextent[0]),
                                   (self.zextent[1] - self.zextent[0])])
        self.shift = torch.tensor([self.xextent[0],
                                   self.yextent[0],
                                   self.zextent[0]])

        self.area = (xextent[1] - xextent[0]) * (yextent[1] - yextent[0])
        self.emitter_av = self.density * self.area
        """Manually override emitter_maximum."""
        if emitter_av is not None:
            self.emitter_av = emitter_av

    def pop(self):
        n = np.random.poisson(lam=self.emitter_av)

        xyz = torch.rand((n, 3)) * self.scale + self.shift
        phot = torch.randint(*self.photon_range, (n, ))
        frame_ix = torch.zeros_like(phot)

        return EmitterSet(xyz=xyz,
                          phot=phot,
                          frame_ix=frame_ix,
                          id=None)


class EmitterPopperMultiFrame(EmitterPopper):
    def __init__(self, xextent, yextent, zextent, density, photon_range, lifetime, num_frames=3, emitter_av=None):
        super().__init__(xextent, yextent, zextent, density, photon_range, emitter_av)
        self.num_frames = num_frames
        self.lifetime = lifetime
        self.lifetime_dist = Exponential(self.lifetime)

        """Determine the number of emitters. Depends on lifetime and num_frames. Rough estimate."""
        self.emitter_av = math.ceil(self.emitter_av * 1.8 * self.num_frames / (0.5 * self.lifetime + 1))

    def pop(self):
        """Pop a multi_frame emitter set."""
        n = np.random.poisson(lam=self.emitter_av)
        xyz = torch.rand((n, 3)) * self.scale + self.shift
        phot = torch.randint(*self.photon_range, (n,))

        """Distribute emitters in time. Increase the range a bit."""
        t0 = torch.rand((n,)) * (self.num_frames + 4 * self.lifetime)
        ontime = self.lifetime_dist.rsample((n, ))

        frame_range = (math.ceil(2 * self.lifetime), math.ceil(2 * self.lifetime) + self.num_frames - 1)
        """Return Emitters with frame index. Use subset of the originally increased range of frames because of
        statistical reasons. Shift index to -1, 0, 1 ..."""
        return LooseEmitterSet(xyz, phot, None, t0, ontime).\
            return_emitterset().get_subset_frame(*frame_range, shift_to=-(self.num_frames - 1)/2)


class RandomPhysical(EmitterGenerator):

    def __init__(self, xextent, yextent, zextent, zsigma, l_exp, num_emitter, act_pd, ep_time):
        """

        :param xextent:     extent, where to place emitters
        :param yextent:
        :param zextent:
        :param num_emitter: number of samples
        :param act_pd:  probability density (dp/dt) of initial activation
        :param ep_time: exposure time in units which correspond to act_pd
        """
        super().__init__()

        self.xextent = xextent
        self.yextent = yextent
        self.zextent = zextent
        self.zsigma = zsigma
        self.l_exp = l_exp

        self.num_emitter = num_emitter
        self.act_pd = act_pd
        self.ep_time = ep_time

        self.total_set = None

    def generate_set(self):
        xyz = torch.rand(self.num_emitter, 3)

        if z_sigma is not None:
            xyz *= torch.tensor([self.xextent[1] - self.xextent[0],
                                 self.yextent[1] - self.yextent[0],
                                 self.zextent[1] - self.zextent[0]])
        else:
            xyz *= torch.tensor([self.xextent[1] - self.xextent[0],
                                 self.yextent[1] - self.yextent[0],
                                 1])
            xyz[:, 2] = (self.zextent[1] + self.zextent[0]) / 2 + torch.randn_like(xyz[:, 2]) * self.zsigma

        phot_total = torch.zeros_like(self.num_emitter).exponential_(self.l_exp)
        frame_ix = torch.ones_like(phot_total) * float('NaN')
        id = torch.arange(0, self.num_emitter)

        self.set_total = EmitterSet(xyz=xyz,
                                    phot=phot_total,
                                    frame_ix=frame_ix,
                                    id=id)
        return self.set_total

    def activation_physics(self, ):
        pass


def emitters_from_csv(csv_file, img_size, cont_radius=3):
    emitters_matlab = pd.read_csv(csv_file)

    emitters_matlab = torch.from_numpy(emitters_matlab.iloc[:, :].as_matrix()).type(torch.float32)
    em_mat = torch.cat((emitters_matlab[:, 2:5] * (img_size[0] + 2 * cont_radius) - cont_radius,  # transform from 0, 1
                        emitters_matlab[:, 5:6],
                        emitters_matlab[:, 1:2] - 1,  # index shift from matlab to python
                        torch.zeros_like(emitters_matlab[:, 0:1])), dim=1)

    warnings.warn('Emitter ID not implemented yet.')

    return split_emitter_cont(em_mat, img_size)


def split_emitter_cont(em_mat, img_size):
    """

    :param em_mat: matrix of all emitters
    :param img_size: img_size in px (not upscaled)
    :return: emitter_matrix and contaminator matrix. contaminators are emitters which are outside the image
    """

    if img_size[0] != img_size[1]:
        raise NotImplementedError("Image must be square at the moment because otherwise the following doesn't work.")

    is_emit = torch.mul((em_mat[:, :2] >= 0).all(1), (em_mat[:, :2] <= img_size[0] - 1).all(1))
    is_cont = ~is_emit

    emit_mat, cont_mat = em_mat[is_emit, :], em_mat[is_cont, :]
    return emit_mat, cont_mat


# https://discuss.pytorch.org/t/efficient-distance-matrix-computation/9065
def pairwise_distances(x, y=None):  # not numerically stable but fast
    '''
    Input: x is a Nxd matrix
           y is an optional Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
            if y is not given then use 'y=x'.
    i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
    '''
    x_norm = (x**2).sum(1).view(-1, 1)
    if y is not None:
        y_norm = (y**2).sum(1).view(1, -1)
    else:
        y = x
        y_norm = x_norm.view(1, -1)

    dist = x_norm + y_norm - 2.0 * torch.mm(x, torch.transpose(y, 0, 1))
    return dist

if __name__ == '__main__':
    extent = ((-0.5, 31.5), (-0.5, 31.5), (-5, 5))
    density = 0.001
    photon_range = (800, 4000)

    runs = torch.zeros(1000)
    for i in range(1000):
        # em = EmitterPopper(extent[0], extent[1], extent[2], density, photon_range, 5).pop()
        em = EmitterPopperMultiFrame(extent[0], extent[1], extent[2], density, photon_range,
                                      lifetime=1, num_frames=3, emitter_av=1).pop()
        em_on_0 = em.split_in_frames(-1, 1)
        em = em_on_0[1]
        runs[i] = em.num_emitter

    import matplotlib.pyplot as plt
    plt.hist(runs)
    plt.show()
    print(runs.mean())

    print("Sucess.")