import warnings
from abc import ABC, abstractmethod  # abstract class

import numpy as np
import time
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

    def __init__(self, *, structure, photon_range: tuple, xy_unit, px_size, density: float = None,
                 emitter_av: float = None):
        super().__init__()

        self.structure = structure
        self._density = density
        self.photon_range = photon_range
        self.xy_unit = xy_unit
        self.px_size = px_size

        """
        Sanity Checks.
        U shall not pa(rse)! (Emitter Average and Density at the same time!
        """
        if (density is None and emitter_av is None) or (density is not None and emitter_av is not None):
            raise ValueError("You must XOR parse either density or emitter average. Not both or none.")

        self.area = self.structure.area

        if emitter_av is not None:
            self._emitter_av = emitter_av
        else:
            self._emitter_av = self._density * self.area

    def pop(self) -> EmitterSet:
        """
        Pop a new EmitterSet

        Returns:
            EmitterSet:

        """
        n = np.random.poisson(lam=self._emitter_av)

        xyz = self.structure.pop(n, 3)
        phot = torch.randint(*self.photon_range, (n,))
        frame_ix = torch.zeros_like(phot)

        return EmitterSet(xyz=xyz,
                          phot=phot,
                          frame_ix=frame_ix,
                          id=None,
                          xy_unit=self.xy_unit,
                          px_size=self.px_size)


class EmitterPopperMultiFrame(EmitterPopperSingle):
    def __init__(self, *, structure, intensity_mu_sig, lifetime, xy_unit, px_size, frames: (int, tuple) = 3,
                 density=None, emitter_av=None, intensity_th=None):
        """

        Args:
            structure:
            intensity_mu_sig:
            lifetime:
            xy_unit:
            px_size:
            frames (int, tuple): if int, it specifies the number of frames, if tuple it specifies the frame range
            density:
            emitter_av:
            intensity_th:
        """
        super().__init__(structure=structure,
                         photon_range=None,
                         xy_unit=xy_unit,
                         px_size=px_size,
                         density=density,
                         emitter_av=emitter_av)

        self.frames = frames
        self.intensity_mu_sig = intensity_mu_sig
        self.intensity_dist = torch.distributions.normal.Normal(self.intensity_mu_sig[0],
                                                                self.intensity_mu_sig[1])
        self.intensity_th = intensity_th if intensity_th is not None else 1e-8
        self.lifetime_avg = lifetime
        self.lifetime_dist = Exponential(1 / self.lifetime_avg)  # parse the rate not the scale ...

        self.t0_dist = torch.distributions.uniform.Uniform(*self.frame_range_plus)

        """
        Determine the total number of emitters. Depends on lifetime and num_frames. 
        Search for the actual value of total emitters on the extended frame range so that on the 0th frame we have
        as many as we have specified in self.emitter_av
        """
        self._emitter_av_total = None
        self._emitter_av_total = self._total_emitter_average_search()

        """Sanity"""
        # if self.num_frames != 3 or self.frame_range != (-1, 1):
        #     warnings.warn("Not yet tested number of frames / frame range.")

    @property
    def frame_range(self):
        if isinstance(self.frames, int):
            return int(-(self.frames - 1) / 2), int((self.frames - 1) / 2)

        elif isinstance(self.frames, (tuple, list)):
            return self.frames

    @property
    def frame_range_plus(self):
        """
        Frame range including buffer in front and end to account for build up effects.

        """
        return self.frame_range[0] - 3 * self.lifetime_avg, self.frame_range[1] + 3 * self.lifetime_avg

    @property
    def num_frames(self):
        if isinstance(self.frames, int):
            return self.frames

        elif isinstance(self.frames, (tuple, list)):
            return self.frames[1] - self.frames[0] + 1

    @property
    def num_frames_plus(self):
        return self.frame_range_plus[1] - self.frame_range_plus[0] + 1

    @classmethod
    def parse(cls, param, structure, frames: (int, tuple)):
        return cls(structure=structure,
                   intensity_mu_sig=param.Simulation.intensity_mu_sig,
                   lifetime=param.Simulation.lifetime_avg,
                   xy_unit=param.Simulation.xy_unit,
                   px_size=param.Camera.px_size,
                   frames=frames,
                   density=param.Simulation.density,
                   emitter_av=param.Simulation.emitter_av,
                   intensity_th=param.Simulation.intensity_th)

    def _test_actual_number(self, num_em) -> int:
        """
        Test actual number of emitters per frame

        Returns:
            int: number of emitters on a target frame
        """
        return len(self.gen_loose_emitter(num_em).return_emitterset().get_subset_frame(*self.frame_range)) / self.num_frames

    def _total_emitter_average_search(self, n: int = 100000):
        """
        Search for the correct total emitter average of loose emitters so that one results in the correct number of
        emitters per frame.

        Args:
            n (int): input samples to test

        Returns:
            number of actual emitters to put in random distribution to get specified number of emitters per frame

        """

        """
        Measure for a significantly large number of emitters and then use rule of proportion to get the correct
        value. An analytical formula would be nice but this is a way to solve the problem ...
        """

        out = self._test_actual_number(n)
        return n / out * self._emitter_av

    def gen_loose_emitter(self, num_em):
        """
        Generate loose EmitterSet

        Returns:
            LooseEmitterSet
        """

        n = np.random.poisson(lam=num_em)
        xyz = self.structure.pop(n, 3)

        """Draw from intensity distribution but clamp the value so as not to fall below 0."""
        intensity = torch.clamp(self.intensity_dist.sample((n,)), self.intensity_th)

        """Distribute emitters in time. Increase the range a bit."""
        t0 = self.t0_dist.sample((n,))
        ontime = self.lifetime_dist.rsample((n,))

        return LooseEmitterSet(xyz, intensity, ontime, t0, xy_unit=self.xy_unit, px_size=self.px_size, id=None)

    def pop(self):
        """
        Return sampled EmitterSet in specified frame range.

        Returns:
            EmitterSet
        """

        loose_em = self.gen_loose_emitter(num_em=self._emitter_av_total)
        em = loose_em.return_emitterset()
        return em.get_subset_frame(*self.frame_range)  # because the simulated frame range is larger
