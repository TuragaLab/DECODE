from abc import ABC, abstractmethod

import deprecated
import torch
from sklearn import neighbors
from torch.nn import functional

from deepsmlm.generic import EmitterSet
from deepsmlm.generic.coordinate_trafo import A2BTransform
from deepsmlm.simulation.psf_kernel import PSF, DeltaPSF


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
                 ix_low: int = None, ix_high: int = None):

        raise NotImplementedError

    def forward(self, em: EmitterSet, ix_low: int = None, ix_high: int = None) -> torch.Tensor:
        """

        Args:
            em (EmitterSet): EmitterSet. Defaults to xyz_nm coordinates.
            ix_low (int): lower frame index
            ix_high (int): upper frame index

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
                 ix_low: int, ix_high: int):

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


@deprecated.deprecated("New version UnifiedEmbedding.")
class SpatialEmbedding(DeltaPSF):
    """
    Compute Spatial Embedding.
    """

    def __init__(self, xextent: tuple, yextent: tuple, img_shape: tuple):
        """

        Args:
            xextent (tuple): extent of the frame in x
            yextent (tuple): extent of the frame in y
            img_shape (tuple): img shape
        """
        super().__init__(xextent=xextent, yextent=yextent, img_shape=img_shape)

        """Setup the bin centers x and y"""
        self._bin_ctr_x = (0.5 * (self._bin_x[1] + self._bin_x[0]) - self._bin_x[0] + self._bin_x)[:-1]
        self._bin_ctr_y = (0.5 * (self._bin_y[1] + self._bin_y[0]) - self._bin_y[0] + self._bin_y)[:-1]

        self._offset_max_x = self._bin_x[1] - self._bin_ctr_x[0]
        self._offset_max_y = self._bin_y[1] - self._bin_ctr_y[0]

    def forward_(self, xyz: torch.Tensor, frame_ix: torch.Tensor, ix_low=None, ix_high=None):
        """

        Args:
            xyz: coordinates of size N x (2 or 3)
            frame_ix: (optional) frame index
            ix_low: (optional) lower frame_index, if None will be determined automatically
            ix_high: (optional) upper frame_index, if None will be determined automatically

        Returns:

        """
        xyz, weight, frame_ix, ix_low, ix_high = PSF.forward(self, xyz, None, frame_ix, ix_low, ix_high)

        mask = self._fov_filter.clean_emitter(xyz)
        n_ix, x_ix, y_ix = self.px_search(xyz[mask], frame_ix[mask].long())

        xy_offset = torch.zeros((ix_high - ix_low + 1, 2, *self.img_shape))
        xy_offset[n_ix, 0, x_ix, y_ix] = xyz[mask, 0] - self._bin_ctr_x[x_ix]
        xy_offset[n_ix, 1, x_ix, y_ix] = xyz[mask, 1] - self._bin_ctr_y[y_ix]

        return xy_offset

    def forward(self, em: EmitterSet, ix_low: int = None, ix_high: int = None):
        return self.forward_(em.xyz_px, em.frame_ix, ix_low, ix_high)


@deprecated.deprecated("New version UnifiedEmbedding.")
class SinglePxEmbedding(TargetGenerator):
    """
    Generate binary target and embeddings of coordinates and photons.
    """

    def __init__(self, xextent: tuple, yextent: tuple, img_shape: tuple):
        """

        Args:
            xextent (tuple): extent of the target map in x
            yextent (tuple): extent of the target map in y
            img_shape (tuple): image shape
        """

        super().__init__(unit='px')

        self._delta = DeltaPSF(xextent,
                               yextent,
                               img_shape)

        self._offset = SpatialEmbedding(xextent, yextent, img_shape)

    def forward_(self, xyz: torch.Tensor, phot: torch.Tensor, frame_ix: torch.Tensor,
                 ix_low: int = None, ix_high: int = None):
        """
        Forward through target generator.

        Args:
            xyz (torch.Tensor): coordinates. Dimension N x 3
            phot (torch.Tensor): photon count. Dimension N
            frame_ix (torch.Tensor(int)): frame_index. Dimension N
            ix_low (int): lower bound for frame_ix
            ix_high (int): upper bound for frame_ix

        Returns:
            target (torch.Tensor): Target frames. Dimension F x 5 x H x W, where F are the frames.
        """

        # mask = self._delta._fov_filter.clean_emitter(xyz)
        # xyz, phot, frame_ix = xyz[mask], phot[mask], frame_ix[mask]
        #
        # n_ix, x_ix, y_ix = self._delta.px_search(xyz, frame_ix.long())
        #
        # embedding = torch.zeros((ix_high - ix_low + 1, 5, *self.img_shape))
        # embedding[n_ix, 0, x_ix, y_ix] =

        p_map = self._delta.forward(xyz, torch.ones_like(xyz[:, 0]), frame_ix, ix_low, ix_high)
        phot_map = self._delta.forward(xyz, phot, frame_ix, ix_low, ix_high)
        xy_map = self._offset.forward_(xyz, frame_ix=frame_ix, ix_low=ix_low, ix_high=ix_high)
        z_map = self._delta.forward(xyz, weight=xyz[:, 2], frame_ix=frame_ix, ix_low=ix_low, ix_high=ix_high)

        return torch.cat((p_map.unsqueeze(1),
                          phot_map.unsqueeze(1),
                          xy_map,
                          z_map.unsqueeze(1)), 1)


@deprecated.deprecated("New version UnifiedEmbedding.")
class KernelEmbedding(SinglePxEmbedding):
    """
    Generate a target with ROI wise embedding (kernel).

    Attributes:
        roi_size (int): size of the ROI in which we define a target
    """

    def __init__(self, xextent, yextent, img_shape, roi_size=3):
        super().__init__(xextent, yextent, img_shape)
        self._xextent = xextent
        self._yextent = yextent
        self._img_shape = img_shape

        self.roi_size = roi_size

        """Addition 'kernel' for phot channel, dxyz"""
        dx = (self._xextent[1] - self._xextent[0]) / self._img_shape[0]
        dy = (self._yextent[1] - self._xextent[0]) / self._img_shape[1]
        self._kern_dx = torch.tensor([[dx, dx, dx], [0., 0., 0.], [-dx, -dx, -dx]])
        self._kern_dy = torch.tensor([[dy, 0., -dy], [dy, 0., -dy], [dy, 0., -dy]])

        """Sanity checks."""
        if self.roi_size != 3:
            raise NotImplementedError("Currently only ROI size 3 is implemented and tested.")

    @staticmethod
    def parse(param):
        return KernelEmbedding(xextent=param.Simulation.psf_extent[0],
                               yextent=param.Simulation.psf_extent[1],
                               img_shape=param.Simulation.img_size,
                               roi_size=param.HyperParameter.target_roi_size)

    def _roi_increaser(self, offset_target):
        """

        Args:
            xyz:
            phot:
            frame_ix:

        Returns:

        """

        # zero pad the target in image space so that we don't have border problems
        offset_target_pad = functional.pad(offset_target, [1, 1, 1, 1], mode='constant', value=0.)
        target = torch.zeros_like(offset_target_pad)

        is_emitter = offset_target_pad[0].nonzero()
        # loop over the non-zero elements
        for i in range(is_emitter.size(0)):
            ix_x = slice(is_emitter[i, 0].item() - 1, is_emitter[i, 0].item() + 2)
            ix_y = slice(is_emitter[i, 1].item() - 1, is_emitter[i, 1].item() + 2)

            """
            Leave p channel unchanged. For the other channels increase the target.
            That means increase target in phot and z channel that we replicate the value (roi-wise constant).
            For x and y we make pointers.
            """
            target[1, ix_x, ix_y] = offset_target_pad[1, is_emitter[i, 0], is_emitter[i, 1]]
            target[2, ix_x, ix_y] = self._kern_dx + offset_target_pad[2, is_emitter[i, 0], is_emitter[i, 1]]
            target[3, ix_x, ix_y] = self._kern_dy + offset_target_pad[3, is_emitter[i, 0], is_emitter[i, 1]]
            target[4, ix_x, ix_y] = offset_target_pad[4, is_emitter[i, 0], is_emitter[i, 1]]

        """Set px centres to original value and remove padding."""
        target[:, is_emitter[:, 0], is_emitter[:, 1]] = offset_target_pad[:, is_emitter[:, 0], is_emitter[:, 1]]
        target = target[:, 1:-1, 1:-1]

        return target

    def forward_(self, xyz: torch.Tensor, phot: torch.Tensor, frame_ix: torch.Tensor,
                 ix_low: int = None, ix_high: int = None):

        offset_target = super().forward_(xyz, phot, frame_ix, ix_low, ix_high)
        for i in range(offset_target.size(0)):
            offset_target[i] = self._roi_increaser(offset_target[i])

        return offset_target


@deprecated.deprecated("Tried some time ago but never used. Probably not up to date. Not tested.")
class GlobalOffsetRep(SinglePxEmbedding):
    def __init__(self, xextent, yextent, zextent, img_shape, photon_threshold=None, masked=False):
        super().__init__(xextent, yextent, zextent, img_shape, cat_output=True, photon_threshold=photon_threshold)
        self.xextent = xextent
        self.yextent = yextent
        self.zextent = zextent
        self.img_shape = img_shape
        self.masked = masked

        self.nearest_neighbor = neighbors.KNeighborsClassifier(1, weights='uniform')

        """Grid coordinates"""
        # ToDo: Double check
        x = torch.linspace(xextent[0], xextent[1], steps=img_shape[0] + 1).float()[:-1]
        y = torch.linspace(yextent[0], yextent[1], steps=img_shape[1] + 1).float()[:-1]

        # add half a pixel to get the center
        x += (x[1] - x[0]) / 2
        y += (y[1] - y[0]) / 2
        self.x_grid, self.y_grid = torch.meshgrid(x, y)
        self.xy_list = torch.cat((self.x_grid.contiguous().view(-1, 1), self.y_grid.contiguous().view(-1, 1)), dim=1)

    def assign_emitter(self, x, coordinates=None):
        """
        Assign px to the closest emitter
        :param x: emitter instance
        :return:
        """
        """Fit the emitters. Fit only in 2D"""
        dummy_index = torch.arange(0, x.num_emitter)
        self.nearest_neighbor.fit(x.xyz[:, :2].numpy(), dummy_index)

        """Predict NN for all image coordinates"""
        if coordinates is None:
            pred_ix = self.nearest_neighbor.predict(self.xy_list)
        else:
            pred_ix = self.nearest_neighbor.predict(coordinates)

        return torch.from_numpy(pred_ix)

    def forward_(self, x):
        """

        :param x: emitter instance
        :return: 5 ch output
        """
        p_map = self._delta.forward(x, torch.ones_like(x.phot))
        p_map[p_map > 1] = 1

        """If masked: Do not output phot / dx, dy, dz everywhere but just around an emitter."""
        if self.masked:
            conv_kernel = torch.ones((1, 1, 3, 3))
            photxyz_mask = functional.conv2d(p_map.unsqueeze(0), conv_kernel, padding=1).squeeze(0).squeeze(0)
            photxyz_mask[photxyz_mask >= 1.] = 1

            img_pseudo_extent = ((-0.5, self.img_shape[0] - 0.5), (-0.5, self.img_shape[1] - 0.5))
            # find indices
            mat_indices = photxyz_mask.squeeze(0).nonzero()
            coordinates = A2BTransform(img_pseudo_extent, (self.xextent, self.yextent)).a2b(mat_indices.float())

            pred_ix = self.assign_emitter(x, coordinates)
            pred_image = (-1) * torch.ones((self.img_shape[0], self.img_shape[1]))
            pred_image[photxyz_mask.byte()] = pred_ix.float()

        else:
            pred_ix = self.assign_emitter(x, None)
            pred_image = pred_ix.reshape(self.img_shape[0], self.img_shape[1])
            photxyz_mask = torch.ones_like(pred_image)

        """Calculate l1 distance per dimension to predicted index for all image coordinates"""
        dx = torch.zeros(self.img_shape[0], self.img_shape[1])
        dy = torch.zeros_like(dx)
        z = torch.zeros_like(dx)
        phot_map = torch.zeros_like(dx)

        for i in range(x.num_emitter):
            mask = (pred_image == i)

            dx[mask] = x.xyz[i, 0] - self.x_grid[mask]
            dy[mask] = x.xyz[i, 1] - self.y_grid[mask]
            z[mask] = x.xyz[i, 2]
            phot_map[mask] = x.phot[i]

        target = torch.cat((p_map, phot_map.unsqueeze(0), dx.unsqueeze(0), dy.unsqueeze(0), z.unsqueeze(0)), 0)
        target[1:] *= photxyz_mask.float()
        return target
