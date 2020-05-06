from abc import ABC, abstractmethod

import torch

from deepsmlm.generic import EmitterSet
from deepsmlm.simulation.psf_kernel import DeltaPSF


class TargetGenerator(ABC):
    def __init__(self, xy_unit='px', ix_low=None, ix_high=None):
        """

        Args:
            unit: Which unit to use for target generator.
        """
        super().__init__()
        self.xy_unit = xy_unit
        self.ix_low = ix_low
        self.ix_high = ix_high

    def _filter_forward(self, em, ix_low, ix_high):

        if ix_low is None:
            ix_low = self.ix_low
        if ix_high is None:
            ix_high = self.ix_high

        """Limit the emitters to the frames of interest and shift the frame index to start at 0."""
        em = em.get_subset_frame(ix_low, ix_high, -ix_low)

        return em, ix_low, ix_high

    @abstractmethod
    def forward_(self, xyz: torch.Tensor, phot: torch.Tensor, frame_ix: torch.Tensor,
                 ix_low: int = None, ix_high: int = None) -> torch.Tensor:
        """

        Args:
            xyz: Coordinates
            phot: Photon values
            frame_ix: Frame index
            ix_low: lower frame index bound
            ix_high: upper frame index bound

        Returns:
            torch.Tensor

        """

        raise NotImplementedError

    def forward(self, em: EmitterSet, ix_low: int = None, ix_high: int = None) -> torch.Tensor:
        """

        Args:
            em (EmitterSet): EmitterSet. Defaults to xyz_nm coordinates.
            ix_low (int): lower frame index bound
            ix_high (int): upper frame index bound

        Returns:
            tar (torch.Tensor): Target.

        """

        em, ix_low, ix_high = self._filter_forward(em, ix_low, ix_high)

        if self.xy_unit == 'px':
            return self.forward_(xyz=em.xyz_px, phot=em.phot, frame_ix=em.frame_ix, ix_low=ix_low, ix_high=ix_high)
        elif self.xy_unit == 'nm':
            return self.forward_(xyz=em.xyz_nm, phot=em.phot, frame_ix=em.frame_ix, ix_low=ix_low, ix_high=ix_high)


class UnifiedEmbeddingTarget(TargetGenerator):

    def __init__(self, xextent: tuple, yextent: tuple, img_shape: tuple, roi_size: int, ix_low=None, ix_high=None):
        super().__init__(xy_unit='px', ix_low=ix_low, ix_high=ix_high)

        self._roi_size = roi_size
        self.img_shape = img_shape

        self.mesh_x, self.mesh_y = torch.meshgrid(
            (torch.arange(-(self._roi_size - 1) // 2, (self._roi_size - 1) // 2 + 1),) * 2)

        self._delta_psf = DeltaPSF(xextent=xextent, yextent=yextent, img_shape=img_shape)
        self._bin_ctr_x = (0.5 * (self._delta_psf._bin_x[1] + self._delta_psf._bin_x[0]) - self._delta_psf._bin_x[
            0] + self._delta_psf._bin_x)[:-1]
        self._bin_ctr_y = (0.5 * (self._delta_psf._bin_y[1] + self._delta_psf._bin_y[0]) - self._delta_psf._bin_y[
            0] + self._delta_psf._bin_y)[:-1]

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

    def _get_central_px(self, xyz, batch_ix):
        """Filter first"""
        mask = self._delta_psf._fov_filter.clean_emitter(xyz)
        return mask, self._delta_psf.px_search(xyz[mask], batch_ix[mask])

    def _get_roi_px(self, batch_ix, x_ix, y_ix):
        xx = self.mesh_x.flatten().to(batch_ix.device)
        yy = self.mesh_y.flatten().to(batch_ix.device)
        n_roi = xx.size(0)

        batch_ix_roi = batch_ix.repeat(n_roi)
        x_ix_roi = (x_ix.view(-1, 1).repeat(1, n_roi) + xx.unsqueeze(0)).flatten()
        y_ix_roi = (y_ix.view(-1, 1).repeat(1, n_roi) + yy.unsqueeze(0)).flatten()

        offset_x = (torch.zeros_like(x_ix).view(-1, 1).repeat(1, n_roi) + xx.unsqueeze(0)).flatten()
        offset_y = (torch.zeros_like(y_ix).view(-1, 1).repeat(1, n_roi) + yy.unsqueeze(0)).flatten()

        belongingness = (torch.arange(y_ix.size(0)).to(batch_ix.device).view(-1, 1).repeat(1, n_roi)).flatten()

        """Limit ROIs by frame dimension"""
        mask = (x_ix_roi >= 0) * (x_ix_roi < self.img_shape[0]) * \
               (y_ix_roi >= 0) * (y_ix_roi < self.img_shape[1])

        batch_ix_roi, x_ix_roi, y_ix_roi, offset_x, offset_y, belongingness = batch_ix_roi[mask], x_ix_roi[mask], \
                                                                              y_ix_roi[mask], \
                                                                              offset_x[mask], offset_y[mask], \
                                                                              belongingness[mask]

        return batch_ix_roi, x_ix_roi, y_ix_roi, offset_x, offset_y, belongingness

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

    def forward_(self, xyz: torch.Tensor, phot: torch.Tensor, frame_ix: torch.Tensor,
                 ix_low: int, ix_high: int) -> torch.Tensor:

        """Get index of central bin for each emitter, throw out emitters that are out of the frame."""
        mask, ix = self._get_central_px(xyz, frame_ix)
        xyz, phot, frame_ix = xyz[mask], phot[mask], frame_ix[mask]

        # unpack and convert
        batch_ix, x_ix, y_ix = ix
        batch_ix, x_ix, y_ix = batch_ix.long(), x_ix.long(), y_ix.long()

        """Get the indices of the ROIs"""
        batch_ix_roi, x_ix_roi, y_ix_roi, offset_x, offset_y, id = self._get_roi_px(batch_ix, x_ix, y_ix)

        batch_size = ix_high - ix_low + 1

        target = torch.zeros((batch_size, 5, *self.img_shape))
        target[:, 0] = self.single_px_target(batch_ix, x_ix, y_ix, batch_size)
        target[:, 1] = self.const_roi_target(batch_ix_roi, x_ix_roi, y_ix_roi, phot, id, batch_size)
        target[:, 2:4] = self.xy_target(batch_ix_roi, x_ix_roi, y_ix_roi, xyz[:, :2], id, batch_size)
        target[:, 4] = self.const_roi_target(batch_ix_roi, x_ix_roi, y_ix_roi, xyz[:, 2], id, batch_size)

        return target

    def forward(self, em: EmitterSet, bg: torch.Tensor = None, ix_low: int = None, ix_high: int = None) -> torch.Tensor:
        target = super().forward(em, ix_low=ix_low, ix_high=ix_high)

        if bg is not None:
            target = torch.cat((target, bg.unsqueeze(1)), 1)

        return target
