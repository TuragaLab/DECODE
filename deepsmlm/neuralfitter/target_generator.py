from abc import ABC

import torch

from deepsmlm.evaluation import predict_dist
from deepsmlm.generic import EmitterSet
from deepsmlm.simulation.psf_kernel import DeltaPSF


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

    def _postprocess_output(self, target: torch.Tensor) -> torch.Tensor:
        """Do some simple post-processual steps before return"""

        if self.squeeze_batch_dim:
            return target.squeeze(0)

        return target


class UnifiedEmbeddingTarget(TargetGenerator):

    def __init__(self, xextent: tuple, yextent: tuple, img_shape: tuple, roi_size: int, ix_low=None, ix_high=None,
                 squeeze_batch_dim: bool = False):
        super().__init__(xy_unit='px', ix_low=ix_low, ix_high=ix_high, squeeze_batch_dim=squeeze_batch_dim)

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
        em, ix_low, ix_high = self._filter_forward(em, ix_low, ix_high)
        target = self.forward_(xyz=em.xyz_px, phot=em.phot, frame_ix=em.frame_ix, ix_low=ix_low, ix_high=ix_high)

        if bg is not None:
            target = torch.cat((target, bg.unsqueeze(0).unsqueeze(0)), 1)

        return self._postprocess_output(target)


class JonasTarget(UnifiedEmbeddingTarget):

    def __init__(self, xextent: tuple, yextent: tuple, img_shape: tuple, roi_size: int, rim_max, ix_low=None,
                 ix_high=None, squeeze_batch_dim: bool = False):
        super().__init__(xextent=xextent, yextent=yextent, img_shape=img_shape, roi_size=roi_size, ix_low=ix_low,
                         ix_high=ix_high, squeeze_batch_dim=squeeze_batch_dim)

        self.rim_max = rim_max

    @classmethod
    def parse(cls, param, **kwargs):
        return cls(xextent=param.Simulation.psf_extent[0],
                   yextent=param.Simulation.psf_extent[1],
                   img_shape=param.Simulation.img_size,
                   roi_size=param.HyperParameter.target_roi_size,
                   rim_max=param.HyperParameter.target_doublette_rim,
                   **kwargs)

    @staticmethod
    def p_from_dxy(dx, dy, active_px, border, rim_max):

        def piecewise_prob(x, border, rim_max):
            prob = torch.zeros_like(x)
            prob[x.abs() <= border] = 1.

            ix_in_rim = (x.abs() > border) * (x.abs() <= rim_max)
            # prob[ix_in_rim] = (0.5 - x[ix_in_rim].abs()) / (rim_max - 0.5) + 1
            prob[ix_in_rim] = 1

            return prob

        ix = (dx.abs() <= 0.7) * (dy.abs() <= 0.5) * active_px
        p_x = torch.zeros_like(dx)
        p_x[ix] = piecewise_prob(dx[ix], border, rim_max)

        # y
        ix = (dy.abs() <= 0.7) * (dx.abs() <= 0.5) * active_px
        p_y = torch.zeros_like(dy)
        p_y[ix] = piecewise_prob(dy[ix], border, rim_max)

        return torch.max(p_x, p_y)

    def forward(self, em: EmitterSet, bg: torch.Tensor = None, ix_low: int = None, ix_high: int = None) -> torch.Tensor:
        tar = super().forward(em, bg, ix_low, ix_high)

        # modify probabilities
        active_px = tar[:, 2] != 0
        tar[:, 0] = self.p_from_dxy(tar[:, 2], tar[:, 3], active_px, 0.5, self.rim_max)

        return self._postprocess_output(tar)


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
