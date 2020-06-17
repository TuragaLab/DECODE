from abc import ABC, abstractmethod
from typing import Union

import torch
import torch.nn

import deepsmlm.generic.emitter as emc
import deepsmlm.simulation.psf_kernel as psf_kernel
from . import target_generator


class OneHotInflator:
    r"""
    Converts single hot px to ROI, i.e. inflates :math:`[0\ 0\ 1\ 0\ 0]` to :math:`[0\ 1\ 1\ 1\ 0]`
    The central pixel (the one hot) will always be preserved.

    Attributes:
        roi_size (int): size of inflation
        channels (int, tuple): channels to which the inflation should apply
        overlap_mode (str): overlap mode
    """

    _overlap_modes_all = ('zero', 'mean')

    def __init__(self, roi_size: int, channels, overlap_mode: str = 'zero'):
        """

        Args:
            roi_size (int): size of inflation
            channels (int, tuple): channels to which the inflation should apply
            overlap_mode (str, optional): overlap mode
        """
        self.roi_size = roi_size
        self.channels = channels
        self.overlap_mode = overlap_mode

        self._pad = torch.nn.ConstantPad2d(1, 0.)
        self._rep_kernel = torch.ones((channels, 1, self.roi_size, self.roi_size))

        """Sanity checks"""
        if self.roi_size != 3:
            raise NotImplementedError("Currently only ROI size 3 is implemented and tested.")

        if self.overlap_mode not in self._overlap_modes_all:
            raise NotImplementedError(f"Non supported overlap mode{self.overlap_mode}. Choose among: "
                                      f"{self._overlap_modes_all}")

    def _is_overlap(self, x):
        """
        Checks for every px whether it is going to be overlapped after inflation and returns the count

        Args:
            x:

        Returns:
            (torch.Tensor, torch.Tensor)
            is_overlap: boolean tensor
            xn_count: overlap count

        """
        # x non zero
        xn = torch.zeros_like(x)
        xn[x != 0] = 1.

        xn_count = torch.nn.functional.conv2d(self._pad(xn), self._rep_kernel, groups=self.channels).long()
        # xn_count *= xn
        is_overlap = xn_count >= 2.

        return is_overlap, xn_count

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forwards tensor through inflator and returns inflated result.

        Args:
            x (torch.Tensor): input

        Returns:
            torch.Tensor:   inflated result

        """

        xctr = x.clone()
        input = self._pad(x).clone()
        xrep = torch.nn.functional.conv2d(input, self._rep_kernel, groups=self.channels)
        overlap_mask, overlap_count = self._is_overlap(x)

        if self.overlap_mode == 'zero':
            xrep[overlap_mask] = 0.
        elif self.overlap_mode == 'mean':
            xrep[overlap_mask] /= overlap_count[overlap_mask]

        xrep[xctr != 0] = xctr[xctr != 0]
        return xrep


class WeightGenerator(target_generator.TargetGenerator):
    """
    Abstract weight generator. A weight is something that is to be multiplied by the (non-reduced) loss, i.e. as

    """

    def __init__(self, ix_low: int = None, ix_high: int = None, squeeze_batch_dim: bool = False):
        super().__init__(xy_unit=None, ix_low=ix_low, ix_high=ix_high, squeeze_batch_dim=squeeze_batch_dim)

    @classmethod
    def parse(cls, param):
        """
        Constructs WeightGenerator by parameter variable which will be likely be a namedtuple, dotmap or similiar.

        Args:
            param:

        Returns:
            WeightGenerator: Instance of WeightGenerator child classes.

        """
        raise NotImplementedError

    def check_forward_sanity(self, tar_em: emc.EmitterSet, tar_frames: torch.Tensor):
        """
        Check sanity of forward arguments, raise error otherwise.

        Args:
            tar_em:
            tar_frames:

        """
        if tar_frames.dim() == 4:
            raise ValueError("Unsupported shape of input.")

        if self.squeeze_batch_dim:
            if tar_frames.size(0) != 1:
                raise ValueError("Squeezing batch dim is only allowed if it is singular.")

    @abstractmethod
    def forward(self, tar_em: emc.EmitterSet, tar_frames: torch.Tensor, ix_low: int, ix_high: int) -> torch.Tensor:
        """
        Calculate weight map based on target frames and target emitters.

        Args:
            tar_em (EmitterSet): target EmitterSet
            tar_frames (torch.Tensor): frames of size :math:`((N,),C,H,W)`

        Returns:
            torch.Tensor: Weight mask of size :math:`((N,),D,H,W)` where likely :math:`C=D`

        """
        raise NotImplementedError


class SimpleWeight(WeightGenerator):

    _weight_bases_all = ('const', 'phot')

    def __init__(self, *, xextent: tuple, yextent: tuple, img_shape: tuple, roi_size: int,
                 weight_mode='const', weight_power: float = None, forward_safety: bool = True,
                 ix_low: Union[int, None], ix_high: Union[int, None], squeeze_batch_dim: bool = False):
        """

        Args:
            xextent (tuple): extent in x
            yextent (tuple): extent in y
            img_shape: image shape
            roi_size (int): roi size of the target
            weight_mode (str): constant or phot
            weight_power (float): power factor of the weight
            forward_safety: check sanity of forward arguments
        """
        super().__init__(ix_low=ix_low, ix_high=ix_high, squeeze_batch_dim=squeeze_batch_dim)

        self.roi_size = roi_size
        self.target_equivalent = target_generator.UnifiedEmbeddingTarget(xextent=xextent, yextent=yextent,
                                                                         img_shape=img_shape, roi_size=roi_size,
                                                                         ix_low=ix_low, ix_high=ix_high)
        self.weight_psf = psf_kernel.DeltaPSF(xextent, yextent, img_shape)

        self.weight_mode = weight_mode
        self.weight_power = weight_power if weight_power is not None else 1.0
        self._forward_safety = forward_safety

        self.check_sanity()

    def check_sanity(self):

        """Sanity checks"""
        if self.weight_mode not in self._weight_bases_all:
            raise ValueError(f"Weight base must be in {self._weight_bases_all}.")

        if self.weight_mode == 'const' and self.weight_power != 1.:
            raise ValueError(f"Weight power of {self.weight_power} != 1."
                             f" which does not have an effect for constant weight mode")

    def check_forward_sanity(self, tar_em: emc.EmitterSet, tar_frames: torch.Tensor):
        super().check_forward_sanity(tar_em, tar_frames)

        if tar_frames.size(1) != 6:
            raise ValueError(f"Unsupported channel dimension.")

        if self.weight_mode != 'const':
            raise NotImplementedError

    def forward(self, tar_em: emc.EmitterSet, tar_frames: torch.Tensor,
                ix_low: Union[int, None], ix_high: Union[int, None]) -> torch.Tensor:

        if self._forward_safety:
            self.check_forward_sanity(tar_em, tar_frames)

        tar_em = self.target_equivalent._filter_forward(tar_em, ix_low, ix_high)

        """Set Detection and Background to 1."""
        weight_frames = torch.zeros_like(tar_frames)
        weight_frames[:, [0, -1]] = 1.

        """Get ROI px set them to the specified weight and rm overlap regions but preserve central px"""
        xyz = tar_em.xyz_px
        ix_batch = tar_em.frame_ix
        weight = torch.ones_like(tar_em.phot)

        batch_size = ix_high - ix_low + 1
        ix_x, ix_y = self.weight_psf.search_bin_index(xyz[:, :2])
        ix_batch_roi, ix_x_roi, ix_y_roi, _, _, id = self.target_equivalent._get_roi_px(ix_batch, ix_x, ix_y)

        """Set ROI"""
        roi_frames = self.target_equivalent.const_roi_target(ix_batch_roi, ix_x_roi, ix_y_roi, weight,
                                                                         id, batch_size)

        """RM overlap but preserve central pixels"""
        ix_roi_unique, roi_count = torch.stack((ix_batch_roi, ix_x_roi, ix_y_roi), 1).unique(dim=0, return_counts=True)
        ix_overlap = roi_count >= 2

        roi_frames[ix_roi_unique[ix_overlap, 0], ix_roi_unique[ix_overlap, 1], ix_roi_unique[ix_overlap, 2]] = 0

        """Preserve central pixels"""
        roi_frames[ix_batch, ix_x, ix_y] = weight

        weight_frames[:, 1:-1] = roi_frames

        return weight_frames


class _SimpleWeight(WeightGenerator):
    """
    Weight mask that is 1 in the detection and background channel everywhere and in the ROIs of the other detection
    channels. Assumes the following channel order prob (0), phot (1), x (2), y (3), z (4), bg (5).

    """

    _weight_bases_all = ('const', 'phot')

    def __init__(self, *, xextent: tuple, yextent: tuple, img_shape: tuple, target_roi_size: int,
                 weight_mode='const', weight_power: float = None):
        """

        Args:
            xextent (tuple): extent in x
            yextent (tuple): extent in y
            img_shape: image shape
            target_roi_size (int): roi size of the target
            weight_mode (str): constant or phot
            weight_power (float): power factor of the weight
        """
        super().__init__()

        self.target_roi_size = target_roi_size
        self.weight_psf = psf_kernel.DeltaPSF(xextent, yextent, img_shape)
        self.delta2roi = OneHotInflator(roi_size=self.target_roi_size,
                                        channels=4,
                                        overlap_mode='zero')

        self.weight_mode = weight_mode
        self.weight_power = weight_power if weight_power is not None else 1.0
        self._forward_safety = True  # safety checks in every forward pass

        """Sanity checks"""
        if self.weight_mode not in self._weight_bases_all:
            raise ValueError(f"Weight base must be in {self._weight_bases_all}.")

        if self.weight_mode == 'const' and self.weight_power != 1.:
            raise ValueError(f"Weight power of {self.weight_power} != 1."
                             f" which does not have an effect for constant weight mode")

    @classmethod
    def parse(cls, param):
        return cls(xextent=param.Simulation.psf_extent[0], yextent=param.Simulation.psf_extent[1],
                   img_shape=param.Simulation.img_size, target_roi_size=param.HyperParameter.target_roi_size,
                   weight_mode=param.HyperParameter.weight_base,
                   weight_power=param.HyperParameter.weight_power)

    def forward(self, tar_frames: torch.Tensor, tar_em: emc.EmitterSet, tar_opt) -> torch.Tensor:
        tar_frames = super().forward(tar_frames, None, None)

        """Safety"""
        if self._forward_safety:
            if tar_frames.size(1) not in (5, 6):
                raise ValueError(f"Unsupported frame dimension {tar_frames.size()}. "
                                 f"Expected channel dimension to be 5 or 6.")

            if not tar_frames.size()[-2:] == torch.Size(self.weight_psf.img_shape):
                raise ValueError("Frame shape not according to init")

            if not (tar_em.phot >= 0.).all():
                raise ValueError(f"Photon count must be greater than zero.\nValues: {tar_em.phot}")

            if self.weight_mode == 'phot':
                if (tar_frames[:, [-1]] == 0).any():
                    raise ValueError("bg must all non 0.")

        """Detection and Background channel"""
        weight = torch.zeros_like(tar_frames)
        weight[:, 0] = 1.
        if weight.size(1) == 6:
            weight[:, 5] = 1.

        if len(tar_em) == 0:  # no target emitter can be returned here after basic init of the weight mask
            return self._forward_return_original(weight)

        if self.weight_mode == 'const':
            weight_pxyz = self.weight_psf.forward(tar_em.xyz, torch.ones_like(tar_em.xyz[:, 0]))
            weight[:, 1:5] = weight_pxyz.unsqueeze(1).repeat(1, 4, 1, 1)

        elif self.weight_mode == 'phot':
            """Simple approximation to the CRLB. """
            weight_phot = self.weight_psf.forward(tar_em.xyz, 1 / tar_em.phot ** self.weight_power)
            weight_xyz = self.weight_psf.forward(tar_em.xyz, tar_em.phot ** self.weight_power)
            weight_pxyz = torch.cat((weight_phot, weight_xyz.repeat(3, 1, 1)), 0).unsqueeze(0)
            weight[:, 1:5] = weight_pxyz
            if weight.size(1) == 6:  # weight of background, CRLB approximation similiar to photon
                weight[:, 5] *= 1 / tar_frames[:, 5] ** self.weight_power

        weight[:, 1:5] = self.delta2roi.forward(weight[:, 1:5])
        return self._forward_return_original(weight)  # return in dimensions of input frame


class FourFoldSimpleWeight(WeightGenerator):

    def __init__(self, *, xextent: tuple, yextent: tuple, img_shape: tuple, target_roi_size: int,
                 rim: float, weight_mode='const', weight_power: float = None):
        super().__init__()
        self.rim = rim

        self.ctr = SimpleWeight(xextent=xextent, yextent=yextent, img_shape=img_shape, target_roi_size=target_roi_size,
                                weight_mode=weight_mode, weight_power=weight_power)

        self.half_x = SimpleWeight(xextent=(xextent[0] + 0.5, xextent[1] + 0.5), yextent=yextent, img_shape=img_shape,
                                   target_roi_size=target_roi_size,
                                   weight_mode=weight_mode, weight_power=weight_power)

        self.half_y = SimpleWeight(xextent=xextent, yextent=(yextent[0] + 0.5, yextent[1] + 0.5), img_shape=img_shape,
                                   target_roi_size=target_roi_size,
                                   weight_mode=weight_mode, weight_power=weight_power)

        self.half_xy = SimpleWeight(xextent=(xextent[0] + 0.5, xextent[1] + 0.5),
                                    yextent=(yextent[0] + 0.5, yextent[1] + 0.5), img_shape=img_shape,
                                    target_roi_size=target_roi_size,
                                    weight_mode=weight_mode, weight_power=weight_power)

    @classmethod
    def parse(cls, param):
        return cls(xextent=param.Simulation.psf_extent[0], yextent=param.Simulation.psf_extent[1],
                   rim=param.HyperParameter.target_train_rim,
                   img_shape=param.Simulation.img_size, target_roi_size=param.HyperParameter.target_roi_size,
                   weight_mode=param.HyperParameter.weight_base,
                   weight_power=param.HyperParameter.weight_power)

    @staticmethod
    def _filter_rim(*args, **kwargs):
        import deepsmlm.neuralfitter.target_generator

        return deepsmlm.neuralfitter.target_generator.FourFoldEmbedding._filter_rim(*args, **kwargs)

    def forward(self, tar_frames: torch.Tensor, tar_em: emc.EmitterSet, tar_opt) -> torch.Tensor:
        tar_frames = super().forward(tar_frames, None, None)

        ctr = self.ctr.forward(tar_frames[:, :5],
                               tar_em=tar_em[self._filter_rim(tar_em.xyz, (-0.5, -0.5), self.rim, (1., 1.))],
                               tar_opt=None)

        hx = self.ctr.forward(tar_frames[:, 5:10],
                              tar_em=tar_em[self._filter_rim(tar_em.xyz_px, (0., -0.5), self.rim, (1., 1.))],
                              tar_opt=None)

        hy = self.ctr.forward(tar_frames[:, 10:15],
                              tar_em=tar_em[self._filter_rim(tar_em.xyz_px, (-0.5, 0.), self.rim, (1., 1.))],
                              tar_opt=None)

        hxy = self.ctr.forward(tar_frames[:, 15:20],
                               tar_em=tar_em[self._filter_rim(tar_em.xyz_px, (0., 0.), self.rim, (1., 1.))],
                               tar_opt=None)

        weight = torch.cat((ctr, hx, hy, hxy), 1)
        if tar_frames.size(1) == 21:
            weight = torch.cat((weight, torch.ones_like(tar_frames[:, [20]])), 1)

        return self._forward_return_original(weight)  # return in dimensions of input frame
