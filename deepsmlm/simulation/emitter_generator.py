from abc import ABC, abstractmethod  # abstract class
import math
import numpy as np
import random
from random import randint
import warnings
import torch
from torch.distributions.exponential import Exponential


from deepsmlm.generic.emitter import EmitterSet, LooseEmitterSet


class EmitterPopperABC(ABC):

    def __init__(self):
        super().__init__()

    def __call__(self) -> EmitterSet:
        return self.pop()

    @abstractmethod
    def pop(self) -> EmitterSet:
        raise NotImplementedError


class EmitterPopperSingle(EmitterPopperABC):
    """
    Simple Emitter sampler. Samples emitters from a structure
    """

    def __init__(self, *, structure, photon_range: tuple, xy_unit, density: float = None, emitter_av: float = None):
        super().__init__()

        self.structure = structure
        self.density = density
        self.photon_range = photon_range
        self.xy_unit = xy_unit

        """
        Sanity Checks.
        U shall not pa(rse)! (Emitter Average and Density at the same time!
        """
        if (density is None and emitter_av is None) or (density is not None and emitter_av is not None):
            raise ValueError("You must XOR parse either density or emitter average. Not both or none.")

        if emitter_av is not None:
            self.area = self.structure.area
            self.emitter_av = emitter_av
        else:
            self.area = self.structure.area
            self.emitter_av = self.density * self.area

    def pop(self) -> EmitterSet:
        """
        Pop a new EmitterSet

        Returns:
            EmitterSet:

        """
        n = np.random.poisson(lam=self.emitter_av)

        xyz = self.structure.pop(n, 3)
        phot = torch.randint(*self.photon_range, (n, ))
        frame_ix = torch.zeros_like(phot)

        return EmitterSet(xyz=xyz,
                          phot=phot,
                          frame_ix=frame_ix,
                          id=None,
                          xy_unit=self.xy_unit)


class EmitterPopperMultiFrame(EmitterPopperSingle):
    def __init__(self, *, structure, intensity_mu_sig, lifetime, xy_unit, num_frames=3, density=None, emitter_av=None,
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
                         xy_unit=xy_unit,
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
        xyz = self.structure.pop(n, 3)

        """Draw from intensity distribution but clamp the value so as not to fall below 0."""
        intensity = torch.clamp(self.intensity_dist.sample((n,)), self.intensity_th, None)

        """Distribute emitters in time. Increase the range a bit."""
        t0 = self.t0_dist.sample((n, ))
        ontime = self.lifetime_dist.rsample((n,))

        return LooseEmitterSet(xyz, intensity, ontime, t0, None, xy_unit=self.xy_unit)

    def _test_actual_number(self):
        """
        Function to test what the actual number of emitters on the target frame is. Basically for debugging.
        :return: number of emitters on 0th frame.
        """
        return len(self.gen_loose_emitter().return_emitterset().get_subset_frame(0, 0))

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

        loose_em = self.gen_loose_emitter()

        emset = loose_em.return_emitterset().get_subset_frame(-1, 1)
        emset.xy_unit = 'px'
        return emset
