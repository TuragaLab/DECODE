from abc import ABC, abstractmethod
from typing import Any, Callable, Union, Optional, Iterable

import torch

from .utils import process
from ..emitter.emitter import EmitterSet
from ..emitter import process as process_em
from ..generic import lazy
from ..simulation import psf_kernel


class TargetGenerator(ABC):
    def __init__(
        self,
        ix_low: Optional[int],
        ix_high: Optional[int],
        filter: Optional[
            Union[process_em.EmitterProcess, Iterable[process_em.EmitterProcess]]
        ],
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


class ParameterList:
    def __init__(
        self,
        n_max: int,
        ix_low: Optional[int] = None,
        ix_high: Optional[int] = None,
        ignore_ix: Optional[bool] = False,
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
            ignore_ix: ignore frame ix and put all emitter attributes on a single non-batched
                tensor
            xy_unit: xy unit
        """

        super().__init__()
        self._ix_low = ix_low
        self._ix_high = ix_high
        self._ignore_ix = ignore_ix
        self._n_em = n_max
        self._xy_unit = xy_unit

        if xy_unit not in {"px", "nm"}:
            raise NotImplementedError

    def forward(self, em: EmitterSet) -> tuple[torch.Tensor, torch.BoolTensor]:
        if self._ignore_ix:
            em = em.clone()
            em.frame_ix[:] = 0

            n_frames = 1
        else:
            # frame filter and shift ix to 0
            em = em.get_subset_frame(self._ix_low, self._ix_high, -self._ix_low)

            n_frames = self._ix_high - self._ix_low

        # compute target (i.e. a matrix / tensor in which all params are concatenated)
        tar = torch.ones((n_frames, self._n_em, 4)) * float("nan")
        mask = torch.zeros((n_frames, self._n_em), dtype=torch.bool)

        # set number of active elements per frame
        for i in range(n_frames):
            em_frame = em.iframe[i]
            n_emitter = len(em_frame)

            if n_emitter > self._n_em:
                raise ValueError(
                    "Number of actual emitters exceeds number of max. emitters."
                )

            mask[i, :n_emitter] = True
            tar[i, :n_emitter, 0] = em_frame.phot
            tar[i, :n_emitter, 1:] = getattr(em_frame, f"xyz_{self._xy_unit}")

        if self._ignore_ix:
            tar.squeeze_(0)
            mask.squeeze_(0)

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
        ix_low: Optional[int],
        ix_high: Optional[int],
        ignore_ix: Optional[bool] = False,
        xy_unit: str = "px",
        filter: Optional[
            Union[process_em.EmitterProcess, Iterable[process_em.EmitterProcess]]
        ] = None,
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
            ignore_ix: ignore frame ix and put all emitters on a single tensor
            xy_unit: xy unit used for target
            filter: emitter filter, forward must take emitters and return emitters
            scaler: scaling, forward must take tensor and return tensor
            switch: disabler to switch off certain attributes, forward must take tensor
                and return tensor
            aux_lane: background processing
        """
        super().__init__(ix_low=ix_low, ix_high=ix_high, filter=filter, scaler=scaler)

        self._list_impl = ParameterList(
            n_max=n_max,
            ix_low=ix_low,
            ix_high=ix_high,
            ignore_ix=ignore_ix,
            xy_unit=xy_unit,
        )
        self._switch = switch
        self._bg_lane = aux_lane

    def forward(
        self, em: EmitterSet, aux: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, torch.BoolTensor, torch.Tensor]:

        em = self._filter_emitters(em)
        tar_list, tar_mask = self._list_impl.forward(em)

        if self._switch is not None:
            tar_list = self._switch.forward(tar_list)

        tar_list = self._scale(tar_list)

        if self._bg_lane is not None:
            aux = self._bg_lane.forward(aux)

        return tar_list, tar_mask, aux


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


class EmbeddingTarget(TargetGenerator):
    def __init__(
        self,
        xextent: tuple[float, float],
        yextent: tuple[float, float],
        img_shape: tuple[int, int],
        roi_size: int,
        ix_low: int,
        ix_high: int,
    ):
        """
        Embedding target, meaning that emitter properties are embedded in pixel space.
        Output is of size Bx5xHxW

        Args:
            xextent:
            yextent:
            img_shape:
            roi_size:
            ix_low:
            ix_high:
        """
        super().__init__(
            ix_low=ix_low,
            ix_high=ix_high,
            filter=[
                process_em.EmitterFilterFrame(
                    ix_low=ix_low,
                    ix_high=ix_high,
                    shift=-ix_low,  # to establish reference starting at 0
                ),
                process_em.EmitterFilterFoV(
                    xextent=xextent,
                    yextent=yextent,
                    zextent=None,
                    xy_unit="px",
                ),
            ],
            scaler=None,
        )

        self._roi_size = roi_size
        self.img_shape = img_shape
        self.mesh_x, self.mesh_y = torch.meshgrid(
            (torch.arange(-(self._roi_size - 1) // 2, (self._roi_size - 1) // 2 + 1),)
            * 2
        )

        self._delta_psf = psf_kernel.DeltaPSF(
            xextent=xextent,
            yextent=yextent,
            img_shape=img_shape,
        )
        self._bin_ctr_x = self._delta_psf.bin_ctr_x
        self._bin_ctr_y = self._delta_psf.bin_ctr_y

    def _get_roi_px(
        self, batch_ix: torch.LongTensor, x_ix: torch.LongTensor, y_ix: torch.LongTensor
    ):
        """
        For each pixel index (aka bin), get the pixel indices around the center
        (i.e. the ROI)

        Args:
            batch_ix:
            x_ix:
            y_ix:

        Returns:
            - batch_ix
            - ix_x
            - ix_y
            - offset_x
            - offset_y
            - id: which element in input array the roi pixel corresponds to

        """

        # pixel pointer relative to the ROI pixels
        xx = self.mesh_x.flatten().to(batch_ix.device)
        yy = self.mesh_y.flatten().to(batch_ix.device)
        n_roi = xx.size(0)

        # Repeat the indices and add an ID for bookkeeping.
        # The idea here is that for the ix we do 'repeat_interleave' and for the
        # offsets we do repeat, such that they overlap correctly. E.g.
        # 5  5  5  9  9  9 (indices)
        # +1 0  -1 +1 0  -1 (offset)
        # 6  5  4  10 9  8 (final indices)
        batch_ix_roi = batch_ix.repeat_interleave(n_roi)
        x_ix_roi = x_ix.repeat_interleave(n_roi)
        y_ix_roi = y_ix.repeat_interleave(n_roi)
        id = torch.arange(x_ix.size(0)).repeat_interleave(n_roi)

        # repeat offsets accordingly and add
        offset_x = xx.repeat(x_ix.size(0))
        offset_y = yy.repeat(y_ix.size(0))
        x_ix_roi = x_ix_roi + offset_x
        y_ix_roi = y_ix_roi + offset_y

        # imit ROIs by frame dimension
        mask = (
            (x_ix_roi >= 0)
            * (x_ix_roi < self.img_shape[0])
            * (y_ix_roi >= 0)
            * (y_ix_roi < self.img_shape[1])
        )

        batch_ix_roi, x_ix_roi, y_ix_roi, offset_x, offset_y, id = (
            batch_ix_roi[mask],
            x_ix_roi[mask],
            y_ix_roi[mask],
            offset_x[mask],
            offset_y[mask],
            id[mask],
        )

        return batch_ix_roi, x_ix_roi, y_ix_roi, offset_x, offset_y, id

    def single_px_target(
        self,
        batch_ix: torch.LongTensor,
        x_ix: torch.LongTensor,
        y_ix: torch.LongTensor,
        batch_size: int,
    ) -> torch.Tensor:
        """
        central pixel target

        Args:
            batch_ix:
            x_ix:
            y_ix:
            batch_size:
        """
        p_tar = torch.zeros((batch_size, *self.img_shape), device=batch_ix.device)
        p_tar[batch_ix, x_ix, y_ix] = 1.0

        return p_tar

    def const_roi_target(
        self,
        batch_ix_roi: torch.LongTensor,
        x_ix_roi: torch.LongTensor,
        y_ix_roi: torch.LongTensor,
        phot: torch.Tensor,
        id: torch.Tensor,
        batch_size: int,
    ) -> torch.Tensor:
        """
        Target for things that are constant within the roi

        Args:
            batch_ix_roi:
            x_ix_roi:
            y_ix_roi:
            phot:
            id:
            batch_size:

        Returns:

        """
        phot_tar = torch.zeros(
            (batch_size, *self.img_shape), device=batch_ix_roi.device
        )
        phot_tar[batch_ix_roi, x_ix_roi, y_ix_roi] = phot[id]

        return phot_tar

    def xy_target(
        self,
        batch_ix_roi: torch.LongTensor,
        x_ix_roi: torch.LongTensor,
        y_ix_roi: torch.LongTensor,
        xy: torch.Tensor,
        id: torch.LongTensor,
        batch_size: int,
    ) -> torch.Tensor:
        """
        XY target.

        Args:
            batch_ix_roi:
            x_ix_roi:
            y_ix_roi:
            xy:
            id:
            batch_size:

        """

        xy_tar = torch.zeros(
            (batch_size, 2, *self.img_shape), device=batch_ix_roi.device
        )
        xy_tar[batch_ix_roi, 0, x_ix_roi, y_ix_roi] = (
            xy[id, 0] - self._bin_ctr_x[x_ix_roi]
        )
        xy_tar[batch_ix_roi, 1, x_ix_roi, y_ix_roi] = (
            xy[id, 1] - self._bin_ctr_y[y_ix_roi]
        )

        return xy_tar

    def _forward_impl(
        self,
        xyz: torch.Tensor,
        phot: torch.Tensor,
        frame_ix: torch.LongTensor,
    ) -> torch.Tensor:
        """
        Compute target

        Args:
            xyz: coordinates
            phot: photon count
            frame_ix: frame_ix with reference at 0
        """
        x_ix, y_ix = self._delta_psf.search_bin_index(xyz[:, :2])

        if not isinstance(frame_ix, torch.LongTensor):
            raise TypeError("Frame index must be integer type.")

        # get the indices of the ROIs around the ctrl pixel
        batch_ix_roi, x_ix_roi, y_ix_roi, offset_x, offset_y, id = self._get_roi_px(
            frame_ix, x_ix, y_ix
        )

        batch_size = self._ix_high - self._ix_low

        target = torch.zeros((batch_size, 5, *self.img_shape))
        target[:, 0] = self.single_px_target(frame_ix, x_ix, y_ix, batch_size)
        target[:, 1] = self.const_roi_target(
            batch_ix_roi, x_ix_roi, y_ix_roi, phot, id, batch_size
        )
        target[:, 2:4] = self.xy_target(
            batch_ix_roi, x_ix_roi, y_ix_roi, xyz[:, :2], id, batch_size
        )
        target[:, 4] = self.const_roi_target(
            batch_ix_roi, x_ix_roi, y_ix_roi, xyz[:, 2], id, batch_size
        )

        return target

    def forward(
        self,
        em: EmitterSet,
        aux: Optional[torch.Tensor] = None,
    ) -> tuple[[torch.Tensor, torch.BoolTensor], torch.Tensor]:

        em = self._filter_emitters(em)

        target = self._forward_impl(
            xyz=em.xyz_px,
            phot=em.phot,
            frame_ix=em.frame_ix,
        )

        return target, aux
