from abc import ABC, abstractmethod

import torch
from sklearn import neighbors
from torch.nn import functional

from deepsmlm.generic import EmitterSet
from deepsmlm.generic.coordinate_trafo import A2BTransform
from deepsmlm.generic.psf_kernel import DeltaPSF, OffsetPSF


class TargetGenerator(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, x_in, weight):
        """

        :param x: input. Will be usually instance of emitterset.
        :return: target
        """
        if isinstance(x_in, EmitterSet):
            pos = x_in.xyz
            if weight is None:
                weight = x_in.phot
            return pos, weight
        else:
            return x_in, weight


class OffsetRep(TargetGenerator):
    """
    Target generator to generate a 5 channel target.
    0th ch: emitter prob
    1st ch: photon value
    2nd ch: dx subpixel offset
    3rd ch: dy subpixel offset
    4th ch: z values
    """
    def __init__(self, xextent, yextent, zextent, img_shape, cat_output=True, photon_threshold=None):
        super().__init__()
        if photon_threshold is not None:
            raise ValueError("Photon threshold not supported within here.")
        self.delta = DeltaPSF(xextent,
                              yextent,
                              zextent,
                              img_shape,
                              photon_normalise=False,
                              photon_threshold=photon_threshold)

        # this might seem as a duplicate, but we need to make sure not to use a photon threshold for generating the z map.
        self.delta_z = DeltaPSF(xextent,
                              yextent,
                              zextent,
                              img_shape,
                              photon_normalise=False,
                              photon_threshold=None)

        self.offset = OffsetPSF(xextent, yextent, img_shape)
        self.cat_out = cat_output

    def forward(self, x, weight=None):
        """
        Create 5 channel output, decode_like
        :param x: emitter instance
        :return: concatenated maps, or single maps with 1 x H x W each (channel order: p, I, x  y, z)
        """
        if weight is None:
            weight = torch.ones_like(x.phot)

        p_map = self.delta.forward(x, weight)
        """It might happen that we see that two emitters are in the same px. p_map will then be 2 or greater.
        As the offset map allows for only one emitter, set the greater than 1 px to 1."""
        p_map[p_map > 1] = 1

        phot_map = self.delta.forward(x, x.phot)
        xy_map = self.offset.forward(x)
        z_map = self.delta_z.forward(x, x.xyz[:, 2])

        if self.cat_out:
            return torch.cat((p_map, phot_map, xy_map, z_map), 0)
        else:
            return p_map, phot_map, xy_map[[0]], xy_map[[1]], z_map


class ROIOffsetRep(OffsetRep):
    """
    Generate a target with increased size as compared to OffsetRep.
    """
    def __init__(self, xextent, yextent, zextent, img_shape, roi_size=3):
        super().__init__(xextent, yextent, zextent, img_shape, cat_output=True, photon_threshold=None)
        self.roi_size = roi_size
        if self.roi_size != 3:
            raise NotImplementedError("Currently only ROI size 3 is implemented and tested.")

        """Addition 'kernel' for phot channel, dxyz"""
        self.kern_dy = torch.tensor([[1., 0., -1.], [1., 0., -1.], [1., 0., -1.]])
        self.kern_dx = torch.tensor([[1., 1., 1], [0., 0., 0.], [-1., -1., -1.]])

    @staticmethod
    def parse(param):
        return ROIOffsetRep(xextent=param.Simulation.psf_extent[0],
                            yextent=param.Simulation.psf_extent[1],
                            zextent=None,
                            img_shape=param.Simulation.img_size,
                            roi_size=param.HyperParameter.target_roi_size)

    def forward(self, x, aux):
        offset_target = super().forward(x)

        """In the photon, dx, dy, dz channel we want to increase the area of information."""
        # zero pad the target in image space so that we don't have border problems
        offset_target_pad = functional.pad(offset_target, (1, 1, 1, 1), mode='constant', value=0.)
        target = torch.zeros_like(offset_target_pad)

        is_emitter = offset_target_pad[0].nonzero()
        # loop over the non-zero elements
        for i in range(is_emitter.size(0)):
            ix_x = slice(is_emitter[i, 0].item() - 1, is_emitter[i, 0].item() + 2)
            ix_y = slice(is_emitter[i, 1].item() - 1, is_emitter[i, 1].item() + 2)

            # leave p_channel unchanged, all others use addition kernel
            target[1, ix_x, ix_y] = offset_target_pad[1, is_emitter[i, 0], is_emitter[i, 1]]
            target[2, ix_x, ix_y] = self.kern_dx + offset_target_pad[2, is_emitter[i, 0], is_emitter[i, 1]]
            target[3, ix_x, ix_y] = self.kern_dy + offset_target_pad[3, is_emitter[i, 0], is_emitter[i, 1]]
            target[4, ix_x, ix_y] = offset_target_pad[4, is_emitter[i, 0], is_emitter[i, 1]]

        # set px centres to original value
        target[:, is_emitter[:, 0], is_emitter[:, 1]] = offset_target_pad[:, is_emitter[:, 0], is_emitter[:, 1]]

        # remove padding
        target = target[:, 1:-1, 1:-1]

        """Add background that was parsed as first auxiliary"""
        target = torch.cat([target, aux[[1]]], 0)

        return target


class GlobalOffsetRep(OffsetRep):
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

    def forward(self, x):
        """

        :param x: emitter instance
        :return: 5 ch output
        """
        p_map = self.delta.forward(x, torch.ones_like(x.phot))
        p_map[p_map > 1] = 1

        """If masked: Do not output phot / dx, dy, dz everywhere but just around an emitter."""
        if self.masked:
            conv_kernel = torch.ones((1, 1, 3, 3))
            photxyz_mask = functional.conv2d(p_map.unsqueeze(0), conv_kernel, padding=1).squeeze(0).squeeze(0)
            photxyz_mask[photxyz_mask >= 1.] = 1

            img_pseudo_extent = ((-0.5, self.img_shape[0]-0.5), (-0.5, self.img_shape[1]-0.5))
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
