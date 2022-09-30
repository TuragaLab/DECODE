from abc import ABC, abstractmethod  # abstract class
from typing import Optional, Union, Protocol

import pydantic
import torch

from . import code
from . import structures
from ..emitter.emitter import EmitterSet, FluorophoreSet
from ..utils import torch as torch_utils


class Sampleable(Protocol):
    def sample(self, n) -> torch.Tensor:
        ...


class _IntUniform:
    @pydantic.validate_arguments
    def __init__(self, low: int, high: int):
        self._low = low
        self._high = high

    def sample(self, n) -> torch.LongTensor:
        return torch.randint(low=self._low, high=self._high, size=(n,))


class EmitterSampler(ABC):
    def __init__(
        self,
        structure: Union[structures.StructurePrior, Sampleable],
        code: Union[code.CodeBook, Sampleable],
        frame_range: Union[int, tuple[int, int]],
        xy_unit,
        px_size,
    ):
        """
        Abstract emitter sampler.
        """
        super().__init__()

        self.structure = structure
        self.code_sampler = code
        # allow int which defaults to (0, frame_range)
        self._frame_range = frame_range if not isinstance(frame_range, int) else (0, frame_range)
        self._xy_unit = xy_unit
        self._px_size = px_size

    @property
    def _n_frames(self) -> int:
        return self._frame_range[1] - self._frame_range[0]

    def __call__(self) -> EmitterSet:
        return self.sample()

    @abstractmethod
    def sample(self) -> EmitterSet:
        raise NotImplementedError


class EmitterSamplerStatic(EmitterSampler):
    def __init__(
        self,
        structure: Union[structures.StructurePrior, Sampleable],
        intensity: Union[tuple, Sampleable],
        em_num: float,
        frame: Union[tuple[int, int], Sampleable],
        frame_range: Union[int, tuple[int, int]],
        code: Optional[Union[code.CodeBook, Sampleable]] = None,
        xy_unit: Optional[str] = None,
        px_size: Optional[tuple[float, float]] = None,
    ):
        """
        Emitter sampler that does not model temporal dynamics.

        Args:
            structure: structure to sample from
            intensity: anything to sample the photon count from or tuple of numbers
                specifying a uniform distribution
            em_num: anything to sample the number of emitters over the frames or number
                specifying the rate of a poisson distribution from which we sample
            frame: anything to sample the frame index from or tuple of frame_ix
                defining lower / upper bound
            frame_range: frame range of the outputted emitters (not necessarily
                equivalent to frame sampler)
            code: anything to sample codes from
            xy_unit:
            px_size:
        """
        super().__init__(
            structure=structure,
            code=code,
            frame_range=frame_range,
            xy_unit=xy_unit,
            px_size=px_size,
        )

        if not hasattr(intensity, "sample"):
            # ToDo: Jonas. Makes sense or rather gaussian?
            intensity = _IntUniform(low=intensity[0], high=intensity[1])

        if not hasattr(frame, "sample"):
            frame = _IntUniform(low=frame[0], high=frame[1])

        self._photon_sampler = intensity
        self._em_num_sampler = None
        self._frame_sampler = frame

        if not hasattr(em_num, "sample"):  # for future purposes
            em_num = torch_utils.ItemizedDist(
                torch.distributions.Poisson(rate=em_num * self._n_frames)
            )

        self._em_num_sampler = em_num

    def sample(self) -> EmitterSet:
        """
        Samples an EmitterSet.
        """
        n = self._em_num_sampler.sample()
        return self.sample_n(n=n)

    @pydantic.validate_arguments
    def sample_n(self, n: int) -> EmitterSet:
        """
        Sample specific number of emitters.

        Args:
            n: number of emitters
        """

        return EmitterSet(
            xyz=self.structure.sample(n),
            phot=self._photon_sampler.sample(n),
            frame_ix=self._frame_sampler.sample(n),
            id=torch.arange(n, dtype=torch.long),
            code=self.code_sampler.sample_codes(n)
            if self.code_sampler is not None
            else None,
            xy_unit=self._xy_unit,
            px_size=self._px_size,
        )


class EmitterSamplerBlinking(EmitterSampler):
    def __init__(
        self,
        *,
        structure: Union[structures.StructurePrior, Sampleable],
        intensity: tuple[float, float],
        em_num: float,
        lifetime: float,
        frame_range: Union[int, tuple[int, int]],
        code: Optional[Union[code.CodeBook, Sampleable]] = None,
        xy_unit: Optional[str] = None,
        px_size: Optional[tuple[float, float]] = None,
    ):
        """
        Emitter sampler that models the blinking dynamics of fluorophores.

        Args:
            structure: structure to sample from
            intensity: anything to sample the photon count from or tuple of numbers
                specifying a uniform distribution
            em_num: anything to sample the number of emitters over the frames or number
                specifying the rate of a poisson distribution from which we sample
            lifetime: average lifetime of the fluorophores
            frame_range: frame range of the outputted emitters (not necessarily
                equivalent to frame sampler)
            code: anything to sample codes from
            xy_unit:
            px_size:
        """

        super().__init__(
            structure=structure,
            code=code,
            frame_range=frame_range,
            xy_unit=xy_unit,
            px_size=px_size,
        )

        self._em_avg = em_num
        self._lifetime = lifetime

        intensity = torch.distributions.Normal(intensity[0], intensity[1])

        self._flux_sampler = intensity
        self._fluo_num_sampler = torch_utils.ItemizedDist(
            torch.distributions.Poisson(rate=self._em_avg_total)
        )
        self._t0_sampler = torch.distributions.uniform.Uniform(*self._time_buffered)
        self._lifetime_sampler = torch.distributions.Exponential(rate=1 / lifetime)

    @property
    def _time_buffered(self) -> tuple[float, float]:
        # time (lower / upper) including buffer by lifetime

        lower = self._frame_range[0] - 3 * self._lifetime
        # upper +2 because + 1 already due to pythonic integer access
        # conferting to float
        upper = self._frame_range[1] + 2 * self._lifetime

        return lower, upper

    @property
    def _duration_buffered(self) -> float:
        # duration (in frame units) including buffer
        return self._time_buffered[1] - self._time_buffered[0]

    @property
    def _em_avg_total(self) -> float:
        # the total number of emitters. Depends on lifetime,
        # frames and emitters. (lifetime + 1) because of binning effect.
        return self._em_avg * self._duration_buffered / (self._lifetime + 1)

    def sample(self) -> EmitterSet:
        n = self._fluo_num_sampler()
        fluo = self.sample_n(n)

        em = fluo.frame_bucketize()
        em = em.get_subset_frame(*self._frame_range)

        return em

    @pydantic.validate_arguments
    def sample_n(self, n: int) -> FluorophoreSet:
        """
        Sample fluorophoreset, i.e. emitters with float time appearance.

        Args:
            n: number of fluorophores
        """
        return FluorophoreSet(
            xyz=self.structure.sample(n),
            flux=self._flux_sampler.sample((n, )).clamp(min=0.),
            ontime=self._lifetime_sampler.sample((n,)),
            t0=self._t0_sampler.sample((n,)),
            id=torch.arange(n, dtype=torch.long),
            code=self.code_sampler.sample(n) if self.code_sampler is not None else None,
            xy_unit=self._xy_unit,
            px_size=self._px_size,
        )
