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

    def __init__(self, structure, density, photon_range, emitter_av=None):
        self.structure = structure
        self.density = density
        self.photon_range = photon_range

        """Manually override emitter number when provided. Area is not needed then."""
        if emitter_av is not None:
            self.area = self.structure.get_area
            self.emitter_av = emitter_av
        else:
            self.area = self.structure.get_area
            self.emitter_av = self.density * self.area

    def pop(self):
        """
        Pop a new sample.
        :return: emitter set
        """
        """If emitter average is set to 0, always pop exactly one single emitter."""
        if self.emitter_av == 0:
            n = 1
        else:
            n = np.random.poisson(lam=self.emitter_av)

        xyz = self.structure.draw(n, 3)
        phot = torch.randint(*self.photon_range, (n, ))
        frame_ix = torch.zeros_like(phot)

        return EmitterSet(xyz=xyz,
                          phot=phot,
                          frame_ix=frame_ix,
                          id=None)


class EmitterPopperMultiFrame(EmitterPopper):
    def __init__(self, structure, density, intensity_mu_sig, lifetime, num_frames=3, emitter_av=None):
        """

        :param structure: structure to sample locations frame
        :param density: density of fluophores
        :param intensity_mu_sig: intensity parametrisation for gaussian dist (mu, sig)
        :param lifetime: average lifetime
        :param num_frames: number of frames
        :param emitter_av: average number of emitters (note that this overrides the density)
        """
        super().__init__(structure, density, None, emitter_av)
        self.num_frames = num_frames
        self.intensity_mu_sig = intensity_mu_sig
        self.intensity_dist = torch.distributions.normal.Normal(self.intensity_mu_sig[0],
                                                                self.intensity_mu_sig[1])
        self.lifetime_avg = lifetime
        self.lifetime_dist = Exponential(1 / self.lifetime_avg)  # parse the rate not the scale ...
        if num_frames != 3:
            raise ValueError("Current emitter generator needs to be changed to allow for more than 3 frames.")

        self.frame_range = (int(-(self.num_frames - 1) / 2), int((self.num_frames - 1) / 2))

        self.t0_dist = torch.distributions.uniform.Uniform(self.frame_range[0] - 3 * self.lifetime_avg,
                                                           self.frame_range[1] + 3 * self.lifetime_avg)

        """Determine the number of emitters. Depends on lifetime and num_frames. Rough estimate."""
        # self.emitter_av = math.ceil(self.emitter_av * 2 * self.num_frames / (1))
        self.emitter_av = emitter_av
        self._emitter_av_total = None

        """
        Search for the actual value of total emitters on the extended frame range so that on the 0th frame we have
        as many as we have specified in self.emitter_av
        """
        self._total_emitter_average_search()

    def gen_loose_emitter(self):
        """
        Generate a loose emitterset (float starting time ...)
        :return: isntance of LooseEmitterset
        """
        """Pop a multi_frame emitter set."""
        if self._emitter_av_total is None:
            lam_in = self.emitter_av
        else:
            lam_in = self._emitter_av_total

        n = np.random.poisson(lam=lam_in)
        xyz = self.structure.draw(n, 3)
        """Draw from intensity distribution but clamp the value so as not to fall below 0."""
        intensity = torch.clamp(self.intensity_dist.sample((n,)), 0, None)

        """Distribute emitters in time. Increase the range a bit."""
        t0 = self.t0_dist.sample((n, ))
        ontime = self.lifetime_dist.rsample((n,))

        return LooseEmitterSet(xyz, intensity, None, t0, ontime)

    def _test_actual_number(self):
        """
        Function to test what the actual number of emitters on the target frame is. Basically for debugging.
        :return: number of emitters on 0th frame.
        """
        return self.gen_loose_emitter().return_emitterset().get_subset_frame(0, 0).num_emitter

    def _total_emitter_average_search(self):
        """
        Search for the correct total emitter average (since the users specifies it on the 0th frame but we have more)
        :return:
        """
        actual_emitters = torch.zeros(20)
        for i in range(actual_emitters.shape[0]):
            actual_emitters[i] = self._test_actual_number()
        actual_emitters = actual_emitters.mean().item()
        self._emitter_av_total = self.emitter_av ** 2 / actual_emitters

    def pop(self):
         """
         Return Emitters with frame index. Use subset of the originally increased range of frames because of
         statistical reasons. Shift index to -1, 0, 1 ...
         :return EmitterSet
         """
         frame_range = (math.ceil(2 * self.lifetime_avg), math.ceil(2 * self.lifetime_avg) + self.num_frames - 1)
         loose_em = self.gen_loose_emitter()

         return loose_em.return_emitterset().get_subset_frame(-1, 1)


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
    Split emitter set into a matrix of emitters and "contaminators". The latter are the ones which are outside the FOV.

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