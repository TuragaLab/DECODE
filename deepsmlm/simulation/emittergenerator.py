from abc import ABC, abstractmethod  # abstract class
import math
import numpy as np
import random
from random import randint
import warnings
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

    def __init__(self, structure, photon_range, density=None, emitter_av=None):
        self.structure = structure
        self.density = density
        self.photon_range = photon_range

        """U shall not pa(rse)! (Emitter Average and Density at the same time!"""
        if (density is None and emitter_av is None) or (density is not None and emitter_av is not None):
            raise ValueError("You must XOR parse either density or emitter average. Not both or none.")

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
    def __init__(self, structure, intensity_mu_sig, lifetime, num_frames=3, density=None, emitter_av=None,
                 intensity_th=None):
        """

        :param structure: structure to sample locations frame
        :param density: density of fluophores
        :param intensity_mu_sig: intensity parametrisation for gaussian dist (mu, sig)
        :param lifetime: average lifetime
        :param num_frames: number of frames
        :param emitter_av: average number of emitters (note that this overrides the density)
        :param intensity_th: defines the minimal intensity
        """
        super().__init__(structure=structure,
                         photon_range=None,
                         density=density,
                         emitter_av=emitter_av)

        self.num_frames = num_frames
        self.intensity_mu_sig = intensity_mu_sig
        self.intensity_dist = torch.distributions.normal.Normal(self.intensity_mu_sig[0],
                                                                self.intensity_mu_sig[1])
        self.intensity_th = intensity_th if intensity_th is not None else 0.0
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
        intensity = torch.clamp(self.intensity_dist.sample((n,)), self.intensity_th, None)

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
        actual_emitters = torch.zeros(50)
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

        emset =  loose_em.return_emitterset().get_subset_frame(-1, 1)
        emset.xy_unit = 'px'
        return emset


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
    import matplotlib.pyplot as plt
    from deepsmlm.simulation.structure_prior import RandomStructure
    extent = ((-0.5, 31.5), (-0.5, 31.5), (-5, 5))
    structure = RandomStructure(*extent)

    runs = torch.zeros(1000)
    for i in range(runs.numel()):

        em = EmitterPopperMultiFrame(structure=structure, density=None, intensity_mu_sig=[10000., 500.],
                                      lifetime=1, num_frames=3, emitter_av=15).pop()
        em_on_0 = em.split_in_frames(-1, 1)
        em = em_on_0[1]
        runs[i] = em.num_emitter

    plt.figure()
    plt.hist(runs)
    plt.show()
    print(runs.mean())

    print("Sucess.")