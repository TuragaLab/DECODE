from abc import ABC, abstractmethod
from typing import Any, Callable, Optional, Union

import torch

from ...emitter import emitter


class ToEmitter(ABC):
    def __init__(
        self,
        xy_unit: Optional[str] = None,
        px_size: Optional[tuple[float, float]] = None,
    ):
        """
        Transforms any into emitters, e.g. model output.
        """
        self._xy_unit = xy_unit
        self._px_size = px_size

    @abstractmethod
    def forward(self, x: Any) -> emitter.EmitterSet:
        raise NotImplementedError


class ToEmitterEmpty(ToEmitter):
    def forward(self, x: Optional[Any] = None) -> emitter.EmitterSet:
        return emitter.factory(0)


class ToEmitterLookUpPixelwise(ToEmitter):
    def __init__(
        self,
        mask: Union[float, Callable[..., torch.BoolTensor]],
        xy_unit: str,
        px_size: Optional[tuple[float, float]] = None,
        pphotxyzbg_mapping: Union[list, tuple] = (0, 1, 2, 3, 4, -1),
        photxyz_sigma_mapping: Union[list, tuple, None] = (5, 6, 7, 8),
        device="cpu",
    ):
        """
        Simple frame to emitter processing in which we threshold the probability output
        (or use a callable to generate a mask)
        and then look-up the features in the respective channels.

        Args:
            mask: raw threshold for filtering or callable that returns boolean tensor
            xy_unit: xy unit
            px_size: pixel size
            pphotxyzbg_mapping: channel index mapping of detection (p), photon, x, y, z, bg
        """
        super().__init__(xy_unit=xy_unit, px_size=px_size)

        self._mask_impl = mask
        self._pphotxyzbg_mapping = pphotxyzbg_mapping
        self._photxyz_sigma_mapping = photxyz_sigma_mapping
        self._device = device

        assert len(self._pphotxyzbg_mapping) == 6, "Wrong length of mapping."
        if self._photxyz_sigma_mapping is not None:
            assert (
                    len(self._photxyz_sigma_mapping) == 4
            ), "Wrong length of sigma mapping."

    def _mask(self, p: torch.Tensor) -> torch.BoolTensor:
        if not callable(self._mask_impl):
            return p >= self._mask_impl
        else:
            return self._mask_impl(p)

    @staticmethod
    def _lookup_features(
        features: torch.Tensor, active_px: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """


        Args:
            features: size :math:`(N, C, H, W)`
            active_px: size :math:`(N, H, W)`

        Returns:
            torch.Tensor: batch-ix, size :math: `M`
            torch.Tensor: extracted features size :math:`(C, M)`

        """

        assert features.dim() == 4
        assert active_px.dim() == features.dim() - 1

        batch_ix = active_px.nonzero(as_tuple=False)[:, 0]
        features_active = features.permute(1, 0, 2, 3)[:, active_px]

        return batch_ix, features_active

    def forward(self, x: torch.Tensor) -> emitter.EmitterSet:
        """
        Forward model output tensor through post-processing and return EmitterSet.
        Will include sigma values in  EmitterSet if mapping was provided initially.

        Args:
            x: tensor from which values are extracted
        """
        # reorder features channel-wise
        x_mapped = x[:, self._pphotxyzbg_mapping]

        active_px = self._mask(x_mapped[:, 0])  # 0th ch. is detection channel
        prob = x_mapped[:, 0][active_px]

        frame_ix, features = self._lookup_features(x_mapped[:, 1:], active_px)
        xyz = features[1:4].transpose(0, 1)

        # if sigma mapping is present, get those values as well
        if self._photxyz_sigma_mapping is not None:
            sigma = x[:, self._photxyz_sigma_mapping]
            _, features_sigma = self._lookup_features(sigma, active_px)

            xyz_sigma = features_sigma[1:4].transpose(0, 1).cpu()
            phot_sigma = features_sigma[0].cpu()
        else:
            xyz_sigma = None
            phot_sigma = None

        return emitter.EmitterSet(
            xyz=xyz.to(self._device),
            frame_ix=frame_ix.to(self._device),
            phot=features[0, :].to(self._device),
            xyz_sig=xyz_sigma,
            phot_sig=phot_sigma,
            bg_sig=None,
            bg=features[4, :].to(self._device) if features.size(0) == 5 else None,
            prob=prob.to(self._device),
            xy_unit=self._xy_unit,
            px_size=self._px_size,
        )


class ToEmitterSpatialIntegration(ToEmitter):
    pass
