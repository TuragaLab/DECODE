from abc import abstractmethod
from deprecated import deprecated
from typing import Union

import torch
import torch.nn

import decode.generic.emitter as emc
import decode.simulation.psf_kernel as psf_kernel
from . import target_generator


class WeightGenerator(target_generator.TargetGenerator):
    """Abstract weight generator. A weight is something that is to be multiplied by the (non-reduced) loss."""

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

    def check_forward_sanity(self, tar_em: emc.EmitterSet, tar_frames: torch.Tensor, ix_low: int, ix_high: int):
        """
        Check sanity of forward arguments, raise error otherwise.

        Args:
            tar_em: target emitters
            tar_frames: target frames
            ix_low: lower frame index
            ix_high: upper frame index

        """
        if tar_frames.dim() != 4:
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
                 ix_low: Union[int, None] = None, ix_high: Union[int, None] = None, squeeze_batch_dim: bool = False):
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

        if self.weight_mode not in self._weight_bases_all:
            raise ValueError(f"Weight base must be in {self._weight_bases_all}.")

        if self.weight_mode == 'const' and self.weight_power != 1.:
            raise ValueError(f"Weight power of {self.weight_power} != 1."
                             f" which does not have an effect for constant weight mode")

    @classmethod
    def parse(cls, param, **kwargs):
        return cls(xextent=param.Simulation.psf_extent[0], yextent=param.Simulation.psf_extent[1],
                   img_shape=param.Simulation.img_size, roi_size=param.HyperParameter.target_roi_size,
                   weight_mode=param.HyperParameter.weight_base,
                   weight_power=param.HyperParameter.weight_power, **kwargs)

    def check_forward_sanity(self, tar_em: emc.EmitterSet, tar_frames: torch.Tensor, ix_low: int, ix_high: int):
        super().check_forward_sanity(tar_em, tar_frames, ix_low, ix_high)

        if tar_frames.size(1) != 6:
            raise ValueError(f"Unsupported channel dimension.")

        if self.weight_mode != 'const':
            raise NotImplementedError

        if (ix_low is not None) or (ix_high is not None):
            if (ix_high - ix_low + 1) != tar_frames.size(0):
                raise ValueError(f"Index does not match")

    def forward(self, tar_em: emc.EmitterSet, tar_frames: torch.Tensor,
                ix_low: Union[int, None] = None, ix_high: Union[int, None] = None) -> torch.Tensor:

        if self.squeeze_batch_dim and tar_frames.dim() == 3:
            tar_frames = tar_frames.unsqueeze(0)

        if self._forward_safety:
            self.check_forward_sanity(tar_em, tar_frames, ix_low, ix_high)

        tar_em, ix_low, ix_high = self.target_equivalent._filter_forward(tar_em, ix_low, ix_high)

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
        if ix_batch_roi.size(0) >= 1:  # this is a Pytorch 1.4 fix
            ix_roi_unique, roi_count = torch.stack((ix_batch_roi, ix_x_roi, ix_y_roi), 1).unique(dim=0,
                                                                                                 return_counts=True)

        else:
            ix_roi_unique = torch.zeros((0, 3)).long()
            roi_count = torch.zeros((0,)).long()

        ix_overlap = roi_count >= 2

        roi_frames[ix_roi_unique[ix_overlap, 0], ix_roi_unique[ix_overlap, 1], ix_roi_unique[ix_overlap, 2]] = 0

        """Preserve central pixels"""
        roi_frames[ix_batch, ix_x, ix_y] = weight

        weight_frames[:, 1:-1] = roi_frames.unsqueeze(1)

        return self._postprocess_output(weight_frames)


@deprecated(reason="Preliminary implementation. Kept if usefuel in future.", version="0.9")
class FourFoldSimpleWeight(WeightGenerator):

    def __init__(self, *, xextent: tuple, yextent: tuple, img_shape: tuple, roi_size: int,
                 rim: float, weight_mode='const', weight_power: float = None):
        super().__init__()
        self.rim = rim

        self.ctr = SimpleWeight(xextent=xextent, yextent=yextent, img_shape=img_shape, roi_size=roi_size,
                                weight_mode=weight_mode, weight_power=weight_power)

        self.half_x = SimpleWeight(xextent=(xextent[0] + 0.5, xextent[1] + 0.5), yextent=yextent, img_shape=img_shape,
                                   roi_size=roi_size,
                                   weight_mode=weight_mode, weight_power=weight_power)

        self.half_y = SimpleWeight(xextent=xextent, yextent=(yextent[0] + 0.5, yextent[1] + 0.5), img_shape=img_shape,
                                   roi_size=roi_size,
                                   weight_mode=weight_mode, weight_power=weight_power)

        self.half_xy = SimpleWeight(xextent=(xextent[0] + 0.5, xextent[1] + 0.5),
                                    yextent=(yextent[0] + 0.5, yextent[1] + 0.5), img_shape=img_shape,
                                    roi_size=roi_size,
                                    weight_mode=weight_mode, weight_power=weight_power)

    @classmethod
    def parse(cls, param):
        return cls(xextent=param.Simulation.psf_extent[0], yextent=param.Simulation.psf_extent[1],
                   rim=param.HyperParameter.target_train_rim,
                   img_shape=param.Simulation.img_size, roi_size=param.HyperParameter.target_roi_size,
                   weight_mode=param.HyperParameter.weight_base,
                   weight_power=param.HyperParameter.weight_power)

    @staticmethod
    def _filter_rim(*args, **kwargs):
        import decode.neuralfitter.target_generator

        return decode.neuralfitter.target_generator.FourFoldEmbedding._filter_rim(*args, **kwargs)

    @staticmethod
    def _add_artfcl_bg(x: torch.Tensor) -> torch.Tensor:
        """
        Add dummy NaN background because we need 6 channels

        Args:
            x:

        Returns:

        """
        assert x.dim() == 4
        assert x.size(1) == 5

        art_bg = float('nan') * torch.ones_like(x[:, [0]])
        return torch.cat((x, art_bg), 1)

    def forward(self, tar_em: emc.EmitterSet, tar_frames: torch.Tensor,
                ix_low: Union[int, None] = None, ix_high: Union[int, None] = None) -> torch.Tensor:
        # tar_frames = super().forward(tar_frames, None, None)

        ctr = self.ctr.forward(tar_em=tar_em[self._filter_rim(tar_em.xyz, (-0.5, -0.5), self.rim, (1., 1.))],
                               tar_frames=self._add_artfcl_bg(tar_frames[:, :5]),
                               ix_low=ix_low, ix_high=ix_high)[:, :-1]

        hx = self.ctr.forward(tar_em=tar_em[self._filter_rim(tar_em.xyz_px, (0., -0.5), self.rim, (1., 1.))],
                              tar_frames=self._add_artfcl_bg(tar_frames[:, 5:10]),
                              ix_low=ix_low, ix_high=ix_high)[:, :-1]

        hy = self.ctr.forward(tar_em=tar_em[self._filter_rim(tar_em.xyz_px, (-0.5, 0.), self.rim, (1., 1.))],
                              tar_frames=self._add_artfcl_bg(tar_frames[:, 10:15]),
                              ix_low=ix_low, ix_high=ix_high)[:, :-1]

        hxy = self.ctr.forward(tar_em=tar_em[self._filter_rim(tar_em.xyz_px, (0., 0.), self.rim, (1., 1.))],
                               tar_frames=self._add_artfcl_bg(tar_frames[:, 15:20]),
                               ix_low=ix_low, ix_high=ix_high)[:, :-1]

        weight = torch.cat((ctr, hx, hy, hxy), 1)
        if tar_frames.size(1) == 21:
            weight = torch.cat((weight, torch.ones_like(tar_frames[:, [20]])), 1)

        return self._postprocess_output(weight)  # return in dimensions of input frame
