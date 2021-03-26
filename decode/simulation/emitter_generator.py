from abc import ABC, abstractmethod  # abstract class
from deprecated import deprecated

import numpy as np
import torch
from torch.distributions.exponential import Exponential

import decode.generic.emitter
from . import structure_prior


class EmitterSampler(ABC):
    """
    Abstract emitter sampler. All implementations / childs must implement a sample method.
    """

    def __init__(self, structure: structure_prior.StructurePrior, xy_unit: str, px_size: tuple):

        super().__init__()

        self.structure = structure
        self.px_size = px_size
        self.xy_unit = xy_unit

    def __call__(self) -> decode.generic.emitter.EmitterSet:
        return self.sample()

    @abstractmethod
    def sample(self) -> decode.generic.emitter.EmitterSet:
        raise NotImplementedError


class EmitterSamplerFrameIndependent(EmitterSampler):
    """
    Simple Emitter sampler. Samples emitters from a structure and puts them all on the same frame, i.e. their
    blinking model is not modelled.

    """

    def __init__(self, *, structure: structure_prior.StructurePrior, photon_range: tuple,
                 density: float = None, em_avg: float = None, xy_unit: str, px_size: tuple):
        """

        Args:
            structure: structure to sample from
            photon_range: range of photon value to sample from (uniformly)
            density: target emitter density (exactly only when em_avg is None)
            em_avg: target emitter average (exactly only when density is None)
            xy_unit: emitter xy unit
            px_size: emitter pixel size

        """

        super().__init__(structure=structure, xy_unit=xy_unit, px_size=px_size)

        self._density = density
        self.photon_range = photon_range

        """
        Sanity Checks.
        U shall not pa(rse)! (Emitter Average and Density at the same time!
        """
        if (density is None and em_avg is None) or (density is not None and em_avg is not None):
            raise ValueError("You must XOR parse either density or emitter average. Not both or none.")

        self.area = self.structure.area

        if em_avg is not None:
            self._em_avg = em_avg
        else:
            self._em_avg = self._density * self.area

    @property
    def em_avg(self) -> float:
        return self._em_avg

    def sample(self) -> decode.generic.emitter.EmitterSet:
        """
        Sample an EmitterSet.

        Returns:
            EmitterSet:

        """
        n = np.random.poisson(lam=self._em_avg)

        return self.sample_n(n=n)

    def sample_n(self, n: int) -> decode.generic.emitter.EmitterSet:
        """
        Sample 'n' emitters, i.e. the number of emitters is given and is not sampled from the Poisson dist.

        Args:
            n: number of emitters

        """

        if n < 0:
            raise ValueError("Negative number of samples is not well-defined.")

        xyz = self.structure.sample(n)
        phot = torch.randint(*self.photon_range, (n,))

        return decode.generic.emitter.EmitterSet(xyz=xyz, phot=phot,
                                                 frame_ix=torch.zeros_like(phot).long(),
                                                 id=torch.arange(n).long(),
                                                 xy_unit=self.xy_unit,
                                                 px_size=self.px_size)


class EmitterSamplerBlinking(EmitterSamplerFrameIndependent):
    def __init__(self, *, structure: structure_prior.StructurePrior, intensity_mu_sig: tuple, lifetime: float,
                 frame_range: tuple, xy_unit: str, px_size: tuple, density=None, em_avg=None, intensity_th=None):
        """

        Args:
            structure:
            intensity_mu_sig:
            lifetime:
            xy_unit:
            px_size:
            frame_range: specifies the frame range
            density:
            em_avg:
            intensity_th:

        """
        super().__init__(structure=structure,
                         photon_range=None,
                         xy_unit=xy_unit,
                         px_size=px_size,
                         density=density,
                         em_avg=em_avg)

        self.n_sampler = np.random.poisson
        self.frame_range = frame_range
        self.intensity_mu_sig = intensity_mu_sig
        self.intensity_dist = torch.distributions.normal.Normal(self.intensity_mu_sig[0],
                                                                self.intensity_mu_sig[1])
        self.intensity_th = intensity_th if intensity_th is not None else 1e-8
        self.lifetime_avg = lifetime
        self.lifetime_dist = Exponential(1 / self.lifetime_avg)  # parse the rate not the scale ...

        self.t0_dist = torch.distributions.uniform.Uniform(*self._frame_range_plus)

        """
        Determine the total number of emitters. Depends on lifetime, frames and emitters.
        (lifetime + 1) because of binning effect.
        """
        self._emitter_av_total = self._em_avg * self._num_frames_plus / (self.lifetime_avg + 1)

    @property
    def _frame_range_plus(self):
        """
        Frame range including buffer in front and end to account for build up effects.

        """
        return self.frame_range[0] - 3 * self.lifetime_avg, self.frame_range[1] + 3 * self.lifetime_avg

    @property
    def num_frames(self):
        return self.frame_range[1] - self.frame_range[0] + 1

    @property
    def _num_frames_plus(self):
        return self._frame_range_plus[1] - self._frame_range_plus[0] + 1

    def sample(self):
        """
        Return sampled EmitterSet in the specified frame range.

        Returns:
            EmitterSet

        """

        n = self.n_sampler(self._emitter_av_total)

        loose_em = self.sample_loose_emitter(n=n)
        em = loose_em.return_emitterset()
        em = em.get_subset_frame(*self.frame_range)  # because the simulated frame range is larger

        return em

    def sample_n(self, *args, **kwargs):
        raise NotImplementedError

    def sample_loose_emitter(self, n) -> decode.generic.emitter.LooseEmitterSet:
        """
        Generate loose EmitterSet. Loose emitters are emitters that are not yet binned to frames.

        Args:
            n: number of 'loose' emitters

        Returns:
            LooseEmitterSet

        """

        xyz = self.structure.sample(n)

        """Draw from intensity distribution but clamp the value so as not to fall below 0."""
        intensity = torch.clamp(self.intensity_dist.sample((n,)), self.intensity_th)

        """Distribute emitters in time. Increase the range a bit."""
        t0 = self.t0_dist.sample((n,))
        ontime = self.lifetime_dist.rsample((n,))

        return decode.generic.emitter.LooseEmitterSet(xyz, intensity, ontime, t0, id=torch.arange(n).long(),
                                                      xy_unit=self.xy_unit, px_size=self.px_size)

    @classmethod
    def parse(cls, param, structure, frames: tuple):
        return cls(structure=structure,
                   intensity_mu_sig=param.Simulation.intensity_mu_sig,
                   lifetime=param.Simulation.lifetime_avg,
                   xy_unit=param.Simulation.xy_unit,
                   px_size=param.Camera.px_size,
                   frame_range=frames,
                   density=param.Simulation.density,
                   em_avg=param.Simulation.emitter_av,
                   intensity_th=param.Simulation.intensity_th)


@deprecated(reason="Deprecated in favour of EmitterSamplerFrameIndependent.", version="0.1.dev")
class EmitterPopperSingle:
    pass


@deprecated(reason="Deprecated in favour of EmitterSamplerBlinking.", version="0.1.dev")
class EmitterPopperMultiFrame:
    pass
