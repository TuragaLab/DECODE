from abc import ABC, abstractmethod
from typing import Any, Callable, Union, Optional

import torch
from deprecated import deprecated

from .utils import process
from ..emitter.emitter import EmitterSet
from ..generic import lazy


class TargetGenerator(ABC):
    def __init__(
        self,
        ix_low: Optional[int],
        ix_high: Optional[int],
        filter: Optional,
        scaler: Optional,
    ):
        """
        Target generator, intended to compute what is input to the loss.

        Args:
            ix_low: lower bound of frame / batch index
            ix_high: upper bound of frame / batch index
            filter: pre_filter
            scaler: scale any to any
        """
        super().__init__()
        self._ix_low = ix_low
        self._ix_high = ix_high

        self._filter = (
            [filter]
            if not (filter is None or isinstance(filter, (list, tuple)))
            else filter
        )
        self._scaler = scaler

    @lazy.no_op_on("_filter")
    def _filter_emitters(self, /, em: EmitterSet) -> EmitterSet:
        for f in self._filter:
            em = f.forward(em)
        return em

    @lazy.no_op_on("_scaler")
    def _scale(self, /, x: Any) -> Any:
        return self._scaler.forward(x)

    def _limit_shift_frames(self, em: EmitterSet) -> EmitterSet:
        # filter emitters to relevant frames and auto-shift index.
        return em.get_subset_frame(self._ix_low, self._ix_high, -self._ix_low)

    @abstractmethod
    def forward(self, em: EmitterSet, aux: Optional[Any] = None) -> Any:
        """
        Forward calculate target as by the emitters and auxiliary input
        (e.g. background).

        Args:
            em: set of emitters
            aux: auxiliaries, likely bg or similiar

        """
        raise NotImplementedError


class ParameterListTarget:
    def __init__(
        self,
        n_max: int,
        ix_low: Optional[int] = None,
        ix_high: Optional[int] = None,
        xy_unit: str = "px",
    ):
        """
        Construct tensor from emitter's attributes.
        The mapping is of size *, 4 with the mapping 0: phtoons, 1: x, 2: y, 3: z

        Args:
            n_max: maximum number of emitters per frame
                (should be much higher than what you draw on average)
            ix_low: lower frame index
            ix_high: upper frame index
            xy_unit: xy unit
        """

        super().__init__()
        self._ix_low = ix_low
        self._ix_high = ix_high
        self._n_frames = ix_high - ix_low
        self._n_em = n_max
        self._xy_unit = xy_unit

        if xy_unit not in {"px", "nm"}:
            raise NotImplementedError

    def forward(self, em: EmitterSet) -> tuple[torch.Tensor, torch.BoolTensor]:
        # frame filter and shift ix to 0
        em = em.get_subset_frame(self._ix_low, self._ix_high, -self._ix_low)

        # compute target (i.e. a matrix / tensor in which all params are concatenated)
        tar = torch.ones((self._n_frames, self._n_em, 4)) * float("nan")
        mask = torch.zeros((self._n_frames, self._n_em), dtype=torch.bool)

        # set number of active elements per frame
        for i in range(self._n_frames):
            em_frame = em.iframe[i]
            n_emitter = len(em_frame)

            if n_emitter > self._n_em:
                raise ValueError(
                    "Number of actual emitters exceeds number of max. emitters."
                )

            mask[i, :n_emitter] = True
            tar[i, :n_emitter, 0] = em_frame.phot
            tar[i, :n_emitter, 1:] = getattr(em_frame, f"xyz_{self._xy_unit}")

        return tar, mask


class DisableAttributes:
    def __init__(self, attr_ix: Union[None, int, tuple, list], val: Union[float] = 0.0):
        """
        Allows to disable attribute prediction of parameter list target;
        e.g. when you don't want to predict z.

        Args:
            attr_ix: index of the attribute you want to disable (phot, x, y, z).
            val: disable value

        """
        self.attr_ix = None
        self.val = val

        # convert to list
        if attr_ix is None or isinstance(attr_ix, (tuple, list)):
            self.attr_ix = attr_ix
        else:
            self.attr_ix = [attr_ix]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.attr_ix is None:
            return x

        x = x.clone()
        x[..., self.attr_ix] = self.val
        return x


class TargetGaussianMixture(TargetGenerator):
    def __init__(
        self,
        n_max: int,
        ix_low: int,
        ix_high: int,
        xy_unit: str = "px",
        filter: Optional[list] = None,
        scaler: Optional = None,
        switch: Optional = None,
        aux_lane: Optional = None,
    ):
        """
        Target for Gaussian Mixture Model loss.

        Args:
            n_max: max number of emitters on a frame
                (should be much higher than average, otherwise an error might be raised).
            ix_low: lower frame ix
            ix_high: upper frame ix
            xy_unit: xy unit used for target
            filter: emitter filter, forward must take emitters and return emitters
            scaler: scaling, forward must take tensor and return tensor
            switch: disabler to switch off certain attributes, forward must take tensor
                and return tensor
            aux_lane: background processing
        """
        super().__init__(ix_low=ix_low, ix_high=ix_high, filter=filter, scaler=scaler)

        self._list_impl = ParameterListTarget(
            n_max=n_max, ix_low=ix_low, ix_high=ix_high, xy_unit=xy_unit
        )
        self._switch = switch
        self._bg_lane = aux_lane

    def forward(
        self, em: EmitterSet, aux: Optional[torch.Tensor] = None
    ) -> tuple[[torch.Tensor, torch.BoolTensor], torch.Tensor]:

        em = self._filter_emitters(em)
        tar_list, tar_mask = self._list_impl.forward(em)

        if self._switch is not None:
            tar_list = self._switch.forward(tar_list)

        tar_list = self._scale(tar_list)

        if self._bg_lane is not None:
            aux = self._bg_lane.forward(aux)

        return (tar_list, tar_mask), aux


class TargetGeneratorChain(TargetGenerator):
    def __init__(self, components: list[TargetGenerator, Any, ...]):
        """
        Chain target generators together to create a new target generator. The first
        component strictly has to be an instance TargetGenerator, the subsequent
        elements can be `Any` but in their forward must accept the output of the
        previous component.

        Args:
            components:
        """
        super().__init__()

        self._components = components
        self._chainer = (
            process.TransformSequence(components[1:]) if len(components) >= 2 else None
        )

    def forward(
        self,
        em: EmitterSet,
        bg: torch.Tensor = None,
        ix_low: Optional[int] = None,
        ix_high: Optional[int] = None,
    ) -> torch.Tensor:

        out = self._components[0].forward(em, bg, ix_low, ix_high)
        if self._chainer is not None:
            out = self._chainer.forward(out)
        return out


class TargetGeneratorFork(TargetGenerator):
    def __init__(
        self,
        components: list[TargetGenerator, ...],
        merger: Optional["TargetGeneratorMerger"] = None,
    ):
        """
        Fork target generators to create a new target generator with parallel
        independent components.

        Args:
            components:
            merger:
        """
        super().__init__()

        merger = merger.forward if merger is not None else None
        self._fork = process.ParallelTransformSequence(
            components, input_slice=None, merger=merger
        )

    def forward(
        self,
        em: EmitterSet,
        bg: torch.Tensor = None,
        ix_low: Optional[int] = None,
        ix_high: Optional[int] = None,
    ) -> list:

        return self._fork.forward(em, bg, ix_low, ix_high)


class TargetGeneratorMerger(TargetGenerator):
    def __init__(self, fn: Callable):
        """
        Helper to merge things after a fork.

        Args:
            fn: merge function
        """
        super().__init__()

        self._fn = fn

    def forward(self, *args: Any) -> Any:
        return self._fn(*args)


class TargetGeneratorForwarder(TargetGenerator):
    _args_valid = {"em", "bg"}

    def __init__(self, args: list[str]):
        super().__init__()
        self._args = args

        if not set(args).issubset(self._args_valid):
            raise NotImplementedError(
                f"Argument specifiction ({args}) is not valid for " f"forwarding."
            )

    def forward(
        self,
        em: EmitterSet,
        bg: torch.Tensor = None,
        ix_low: Optional[int] = None,
        ix_high: Optional[int] = None,
    ) -> Union[None, EmitterSet, torch.Tensor, tuple[EmitterSet, torch.Tensor]]:

        if len(self._args) == 0:
            return
        elif "em" in self._args and "bg" in self._args:
            return em, bg
        elif "em" in self._args:
            return em
        elif "bg" in self._args:
            return bg

        raise NotImplementedError


@deprecated(version="v0.11", reason="Not in use.")
class UnifiedEmbeddingTarget(TargetGenerator):
    pass
