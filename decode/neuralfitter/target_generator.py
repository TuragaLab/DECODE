from abc import ABC, abstractmethod
from typing import Union

import torch

from decode.evaluation import predict_dist
from decode.generic import EmitterSet
from decode.generic import process
from decode.generic.process import RemoveOutOfFOV
from decode.simulation.psf_kernel import DeltaPSF


class TargetGenerator(ABC):
    def __init__(self, xy_unit='px', ix_low: int = None, ix_high: int = None, squeeze_batch_dim: bool = False):
        """

        Args:
            xy_unit: Which unit to use for target generator
            ix_low: lower bound of frame / batch index
            ix_high: upper bound of frame / batch index
            squeeze_batch_dim: if lower and upper frame_ix are the same, squeeze out the batch dimension before return

        """
        super().__init__()

        self.xy_unit = xy_unit
        self.ix_low = ix_low
        self.ix_high = ix_high
        self.squeeze_batch_dim = squeeze_batch_dim

        self.sanity_check()

    def sanity_check(self):

        if self.squeeze_batch_dim and self.ix_low != self.ix_high:
            raise ValueError(f"Automatic batch squeeze can only be used when upper and lower ix fall together.")

    def _filter_forward(self, em: EmitterSet, ix_low: (int, None), ix_high: (int, None)):
        """
        Filter emitters and auto-set frame bounds

        Args:
            em:
            ix_low:
            ix_high:

        Returns:
            em (EmitterSet): filtered EmitterSet
            ix_low (int): lower frame index
            ix_high (int): upper frame index

        """

        if ix_low is None:
            ix_low = self.ix_low
        if ix_high is None:
            ix_high = self.ix_high

        """Limit the emitters to the frames of interest and shift the frame index to start at 0."""
        em = em.get_subset_frame(ix_low, ix_high, -ix_low)

        return em, ix_low, ix_high

    def _postprocess_output(self, x: torch.Tensor) -> torch.Tensor:
        """
        Some simple post-processual steps before return.

        Args:
            x: input of size :math:`(N,C,H,W)`

        """

        if self.squeeze_batch_dim:
            if x.size(0) != 1:
                raise ValueError("First, batch dimension, not singular.")

            return x.squeeze(0)

        return x

    @abstractmethod
    def forward(self, em: EmitterSet, bg: torch.Tensor = None, ix_low: int = None, ix_high: int = None) -> torch.Tensor:
        """
        Forward calculate target as by the emitters and background. Overwrite the default frame ix boundaries.

        Args:
            em: set of emitters
            bg: background frame
            ix_low: lower frame index
            ix_high: upper frame index

        Returns:
            target frames

        """
        raise NotImplementedError


class UnifiedEmbeddingTarget(TargetGenerator):

    def __init__(self, xextent: tuple, yextent: tuple, img_shape: tuple, roi_size: int, ix_low=None, ix_high=None,
                 squeeze_batch_dim: bool = False):
        super().__init__(xy_unit='px', ix_low=ix_low, ix_high=ix_high, squeeze_batch_dim=squeeze_batch_dim)

        self._roi_size = roi_size
        self.img_shape = img_shape

        self.mesh_x, self.mesh_y = torch.meshgrid(
            (torch.arange(-(self._roi_size - 1) // 2, (self._roi_size - 1) // 2 + 1),) * 2)

        self._delta_psf = DeltaPSF(xextent=xextent, yextent=yextent, img_shape=img_shape)
        self._em_filter = process.RemoveOutOfFOV(xextent=xextent, yextent=yextent, zextent=None, xy_unit='px')
        self._bin_ctr_x = self._delta_psf.bin_ctr_x
        self._bin_ctr_y = self._delta_psf.bin_ctr_y

    @property
    def xextent(self):
        return self._delta_psf.xextent

    @property
    def yextent(self):
        return self._delta_psf.yextent

    @classmethod
    def parse(cls, param, **kwargs):
        return cls(xextent=param.Simulation.psf_extent[0],
                   yextent=param.Simulation.psf_extent[1],
                   img_shape=param.Simulation.img_size,
                   roi_size=param.HyperParameter.target_roi_size,
                   **kwargs)

    def _get_roi_px(self, batch_ix, x_ix, y_ix):
        """
        For each pixel index (aka bin), get the pixel around the center (i.e. the ROI)

        Args:
            batch_ix:
            x_ix:
            y_ix:

        Returns:

        """

        """Pixel pointer relative to the ROI pixels"""
        xx = self.mesh_x.flatten().to(batch_ix.device)
        yy = self.mesh_y.flatten().to(batch_ix.device)
        n_roi = xx.size(0)

        """
        Repeat the indices and add an ID for bookkeeping. 
        The idea here is that for the ix we do 'repeat_interleave' and for the offsets we do repeat, such that they 
        overlap correctly. E.g.
        5  5  5  9  9  9 (indices)
        +1 0  -1 +1 0  -1 (offset)
        6  5  4  10 9  8 (final indices)
        """
        batch_ix_roi = batch_ix.repeat_interleave(n_roi)
        x_ix_roi = x_ix.repeat_interleave(n_roi)
        y_ix_roi = y_ix.repeat_interleave(n_roi)
        id = torch.arange(x_ix.size(0)).repeat_interleave(n_roi)

        """Repeat offsets accordingly and add"""
        offset_x = xx.repeat(x_ix.size(0))
        offset_y = yy.repeat(y_ix.size(0))
        x_ix_roi = x_ix_roi + offset_x
        y_ix_roi = y_ix_roi + offset_y

        """Limit ROIs by frame dimension"""
        mask = (x_ix_roi >= 0) * (x_ix_roi < self.img_shape[0]) * \
               (y_ix_roi >= 0) * (y_ix_roi < self.img_shape[1])

        batch_ix_roi, x_ix_roi, y_ix_roi, offset_x, offset_y, id = batch_ix_roi[mask], x_ix_roi[mask], \
                                                                   y_ix_roi[mask], \
                                                                   offset_x[mask], offset_y[mask], \
                                                                   id[mask]

        return batch_ix_roi, x_ix_roi, y_ix_roi, offset_x, offset_y, id

    def single_px_target(self, batch_ix, x_ix, y_ix, batch_size):
        p_tar = torch.zeros((batch_size, *self.img_shape)).to(batch_ix.device)
        p_tar[batch_ix, x_ix, y_ix] = 1.

        return p_tar

    def const_roi_target(self, batch_ix_roi, x_ix_roi, y_ix_roi, phot, id, batch_size):
        phot_tar = torch.zeros((batch_size, *self.img_shape)).to(batch_ix_roi.device)
        phot_tar[batch_ix_roi, x_ix_roi, y_ix_roi] = phot[id]

        return phot_tar

    def xy_target(self, batch_ix_roi, x_ix_roi, y_ix_roi, xy, id, batch_size):
        xy_tar = torch.zeros((batch_size, 2, *self.img_shape)).to(batch_ix_roi.device)
        xy_tar[batch_ix_roi, 0, x_ix_roi, y_ix_roi] = xy[id, 0] - self._bin_ctr_x[x_ix_roi]
        xy_tar[batch_ix_roi, 1, x_ix_roi, y_ix_roi] = xy[id, 1] - self._bin_ctr_y[y_ix_roi]

        return xy_tar

    def _filter_forward(self, em: EmitterSet, ix_low: (int, None), ix_high: (int, None)):
        """
        Filter as in abstract class, plus kick out emitters that are outside the frame

        Args:
            em:
            ix_low:
            ix_high:

        """
        em, ix_low, ix_high = super()._filter_forward(em, ix_low, ix_high)
        em = self._em_filter.forward(em)  # kick outside of frame out

        return em, ix_low, ix_high

    def forward_(self, xyz: torch.Tensor, phot: torch.Tensor, frame_ix: torch.LongTensor,
                 ix_low: int, ix_high: int) -> torch.Tensor:
        """Get index of central bin for each emitter."""
        x_ix, y_ix = self._delta_psf.search_bin_index(xyz[:, :2])

        assert isinstance(frame_ix, torch.LongTensor)

        """Get the indices of the ROIs around the ctrl pixel"""
        batch_ix_roi, x_ix_roi, y_ix_roi, offset_x, offset_y, id = self._get_roi_px(frame_ix, x_ix, y_ix)

        batch_size = ix_high - ix_low + 1

        target = torch.zeros((batch_size, 5, *self.img_shape))
        target[:, 0] = self.single_px_target(frame_ix, x_ix, y_ix, batch_size)
        target[:, 1] = self.const_roi_target(batch_ix_roi, x_ix_roi, y_ix_roi, phot, id, batch_size)
        target[:, 2:4] = self.xy_target(batch_ix_roi, x_ix_roi, y_ix_roi, xyz[:, :2], id, batch_size)
        target[:, 4] = self.const_roi_target(batch_ix_roi, x_ix_roi, y_ix_roi, xyz[:, 2], id, batch_size)

        return target

    def forward(self, em: EmitterSet, bg: torch.Tensor = None, ix_low: int = None, ix_high: int = None) -> torch.Tensor:
        em, ix_low, ix_high = self._filter_forward(em, ix_low, ix_high)  # filter em that are out of view
        target = self.forward_(xyz=em.xyz_px, phot=em.phot, frame_ix=em.frame_ix, ix_low=ix_low, ix_high=ix_high)

        if bg is not None:
            target = torch.cat((target, bg.unsqueeze(0).unsqueeze(0)), 1)

        return self._postprocess_output(target)


class ParameterListTarget(TargetGenerator):
    def __init__(self, n_max: int, xextent: tuple, yextent: tuple, ix_low=None, ix_high=None, xy_unit: str = 'px',
                 squeeze_batch_dim: bool = False):
        """
        Target corresponding to the Gausian-Mixture Model Loss. Simply cat all emitter's attributes up to a
         maximum number of emitters as a list.

        Args:
            n_max: maximum number of emitters (should be multitude of what you draw on average)
            xextent: extent of the emitters in x
            yextent: extent of the emitters in y
            ix_low: lower frame index
            ix_high: upper frame index
            xy_unit: xy unit
            squeeze_batch_dim: squeeze batch dimension before return
        """

        super().__init__(xy_unit=xy_unit, ix_low=ix_low, ix_high=ix_high, squeeze_batch_dim=squeeze_batch_dim)
        self.n_max = n_max
        self.xextent = xextent
        self.yextent = yextent

        self._fov_filter = RemoveOutOfFOV(xextent=xextent, yextent=yextent, xy_unit=xy_unit)

    def _filter_forward(self, em: EmitterSet, ix_low: (int, None), ix_high: (int, None)):
        em, ix_low, ix_high = super()._filter_forward(em, ix_low, ix_high)
        em = self._fov_filter.forward(em)

        return em, ix_low, ix_high

    def forward(self, em: EmitterSet, bg: torch.Tensor = None, ix_low: int = None, ix_high: int = None):
        em, ix_low, ix_high = self._filter_forward(em, ix_low, ix_high)

        n_frames = ix_high - ix_low + 1

        """Setup and compute parameter target (i.e. a matrix / tensor in which all params are concatenated)."""
        param_tar = torch.zeros((n_frames, self.n_max, 4))
        mask_tar = torch.zeros((n_frames, self.n_max)).bool()

        if self.xy_unit == 'px':
            xyz = em.xyz_px
        elif self.xy_unit == 'nm':
            xyz = em.xyz_nm
        else:
            raise NotImplementedError

        """Set number of active elements per frame"""
        for i in range(n_frames):
            n_emitter = len(em.get_subset_frame(i, i))

            if n_emitter > self.n_max:
                raise ValueError("Number of actual emitters exceeds number of max. emitters.")

            mask_tar[i, :n_emitter] = 1

            ix = em.frame_ix == i
            param_tar[i, :n_emitter, 0] = em[ix].phot
            param_tar[i, :n_emitter, 1:] = xyz[ix]

        return self._postprocess_output(param_tar), self._postprocess_output(mask_tar), bg


class DisableAttributes:

    def __init__(self, attr_ix: Union[None, int, tuple, list]):
        """
        Allows to disable attribute prediction of parameter list target; e.g. when you don't want to predict z.

        Args:
            attr_ix: index of the attribute you want to disable (phot, x, y, z).

        """
        self.attr_ix = None

        # convert to list
        if attr_ix is None or isinstance(attr_ix, (tuple, list)):
            self.attr_ix = attr_ix
        else:
            self.attr_ix = [attr_ix]

    def forward(self, param_tar, mask_tar, bg):
        if self.attr_ix is None:
            return param_tar, mask_tar, bg

        param_tar[..., self.attr_ix] = 0.
        return param_tar, mask_tar, bg

    @classmethod
    def parse(cls, param):
        return cls(attr_ix=param.HyperParameter.disabled_attributes)


class FourFoldEmbedding(TargetGenerator):

    def __init__(self, xextent: tuple, yextent: tuple, img_shape: tuple, rim_size: float,
                 roi_size: int, ix_low=None, ix_high=None, squeeze_batch_dim: bool = False):
        super().__init__(xy_unit='px', ix_low=ix_low, ix_high=ix_high, squeeze_batch_dim=squeeze_batch_dim)

        self.xextent_native = xextent
        self.yextent_native = yextent

        self.rim = rim_size
        self.img_shape = img_shape
        self.roi_size = roi_size

        self.embd_ctr = UnifiedEmbeddingTarget(xextent=xextent, yextent=yextent, img_shape=img_shape,
                                               roi_size=roi_size, ix_low=ix_low, ix_high=ix_high)

        self.embd_half_x = UnifiedEmbeddingTarget(xextent=(xextent[0] + 0.5, xextent[1] + 0.5), yextent=yextent,
                                                  img_shape=img_shape, roi_size=roi_size,
                                                  ix_low=ix_low, ix_high=ix_high)

        self.embd_half_y = UnifiedEmbeddingTarget(xextent=xextent, yextent=(yextent[0] + 0.5, yextent[1] + 0.5),
                                                  img_shape=img_shape, roi_size=roi_size,
                                                  ix_low=ix_low, ix_high=ix_high)

        self.embd_half_xy = UnifiedEmbeddingTarget(xextent=(xextent[0] + 0.5, xextent[1] + 0.5),
                                                   yextent=(yextent[0] + 0.5, yextent[1] + 0.5),
                                                   img_shape=img_shape, roi_size=roi_size,
                                                   ix_low=ix_low, ix_high=ix_high)

    @classmethod
    def parse(cls, param, **kwargs):
        return cls(xextent=param.Simulation.psf_extent[0],
                   yextent=param.Simulation.psf_extent[1],
                   img_shape=param.Simulation.img_size,
                   roi_size=param.HyperParameter.target_roi_size,
                   rim_size=param.HyperParameter.target_train_rim,
                   **kwargs)

    @staticmethod
    def _filter_rim(xy, xy_0, rim, px_size) -> torch.BoolTensor:
        """
        Takes coordinates and checks whether they are close to a pixel border (i.e. within a rim).
        True if not in rim, false if in rim.

        Args:
            xy:
            xy_0:
            rim:
            px_size:

        Returns:

        """

        """Transform coordinates relative to unit px"""
        x_rel = (predict_dist.px_pointer_dist(xy[:, 0], xy_0[0], px_size[0]) - xy_0[0]) / px_size[0]
        y_rel = (predict_dist.px_pointer_dist(xy[:, 1], xy_0[1], px_size[1]) - xy_0[1]) / px_size[1]

        """Falsify coordinates that are inside the rim"""
        ix = (x_rel >= rim) * (x_rel < (1 - rim)) * (y_rel >= rim) * (y_rel < (1 - rim))

        return ix

    def forward(self, em: EmitterSet, bg: torch.Tensor = None, ix_low: int = None, ix_high: int = None) -> torch.Tensor:
        em, ix_low, ix_high = self._filter_forward(em, ix_low=ix_low, ix_high=ix_high)

        """Forward through each and all targets and filter the rim"""
        ctr = self.embd_ctr.forward(em=em[self._filter_rim(em.xyz_px, (-0.5, -0.5), self.rim, (1., 1.))],
                                    bg=None, ix_low=ix_low, ix_high=ix_high)

        half_x = self.embd_half_x.forward(em=em[self._filter_rim(em.xyz_px, (0., -0.5), self.rim, (1., 1.))],
                                          bg=None, ix_low=ix_low, ix_high=ix_high)

        half_y = self.embd_half_y.forward(em=em[self._filter_rim(em.xyz_px, (-0.5, 0.), self.rim, (1., 1.))],
                                          bg=None, ix_low=ix_low, ix_high=ix_high)

        half_xy = self.embd_half_xy.forward(em=em[self._filter_rim(em.xyz_px, (0., 0.), self.rim, (1., 1.))],
                                            bg=None, ix_low=ix_low, ix_high=ix_high)

        target = torch.cat((ctr, half_x, half_y, half_xy), 1)

        if bg is not None:
            target = torch.cat((target, bg.unsqueeze(0).unsqueeze(0)), 1)

        return self._postprocess_output(target)
