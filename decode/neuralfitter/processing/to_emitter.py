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

        self._mask = mask
        self._pphotxyzbg_mapping = pphotxyzbg_mapping
        self._photxyz_sigma_mapping = photxyz_sigma_mapping
        self._device = device

        assert len(self._pphotxyzbg_mapping) == 6, "Wrong length of mapping."
        if self._photxyz_sigma_mapping is not None:
            assert (
                len(self._photxyz_sigma_mapping) == 4
            ), "Wrong length of sigma mapping."

    def _mask_impl(self, p: torch.Tensor) -> torch.BoolTensor:
        if not callable(self._mask):
            return p >= self._mask
        else:
            return self._mask(p)

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

        active_px = self._mask_impl(x_mapped[:, 0])  # 0th ch. is detection channel
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


def _norm_sum(*args):
    return torch.clamp(torch.add(*args), 0.0, 1.0)


class ToEmitterSpatialIntegration(ToEmitterLookUpPixelwise):
    _p_aggregations = ("sum", "norm_sum")  # , 'max', 'pbinom_cdf', 'pbinom_pdf')

    def __init__(
        self,
        raw_th: float,
        xy_unit: str,
        px_size: tuple[float, float] = None,
        pphotxyzbg_mapping: Union[list, tuple] = (0, 1, 2, 3, 4, -1),
        photxyz_sigma_mapping: Union[list, tuple, None] = (5, 6, 7, 8),
        p_aggregation: Union[str, Callable] = "norm_sum",
        _split_th: float = 0.6,
    ):
        """
        Spatial Integration to handle local clusters of non-zero probability.
        The procedure is to cut off low probability predictions, get localizations with
        high probabily which are used as is, but for local clusters find maxima and sum
        up their probabilites.

        Args:
            raw_th: probability threshold from where detections are considered
            xy_unit: unit of the xy coordinates
            px_size: pixel size
            pphotxyzbg_mapping: channel index mapping
            photxyz_sigma_mapping: channel index mapping of sigma channels
            p_aggregation: aggreation method to aggregate probabilities.
             can be 'sum', 'max', 'norm_sum'
            _split_th: threshold above which a prediction is treated as `easy` in that
             it is a clear prediction.
        """
        super().__init__(
            mask=raw_th,
            xy_unit=xy_unit,
            px_size=px_size,
            pphotxyzbg_mapping=pphotxyzbg_mapping,
            photxyz_sigma_mapping=photxyz_sigma_mapping,
        )
        self._raw_th = raw_th
        self._split_th = _split_th
        self._p_aggregation = self._set_p_aggregation(p_aggregation)

    def forward(self, x: torch.Tensor) -> emitter.EmitterSet:
        x[:, 0] = self._non_max_suppression(x[:, 0])
        return super().forward(x)

    def _non_max_suppression(self, p: torch.Tensor) -> torch.Tensor:
        """

        Args:
            p: probability map
        """
        with torch.no_grad():
            p_copy = p.clone()

            # Probability values > 0.3 are regarded as possible locations
            p_clip = torch.where(p > self._raw_th, p, torch.zeros_like(p))[:, None]

            # localize maximum values within a 3x3 patch
            pool = torch.nn.functional.max_pool2d(p_clip, 3, 1, padding=1)
            max_mask1 = torch.eq(p[:, None], pool).float()

            # Add probability values from the 4 adjacent pixels
            diag = 0.0  # 1/np.sqrt(2)
            filt = (
                torch.tensor([[diag, 1.0, diag], [1, 1, 1], [diag, 1, diag]])
                .unsqueeze(0)
                .unsqueeze(0)
                .to(p.device)
            )
            conv = torch.nn.functional.conv2d(p[:, None], filt, padding=1)
            p_ps1 = max_mask1 * conv

            # to identify two fluorophores in adjacent pixels we look
            # for probablity values > 0.6 that are not part of the first mask
            p_copy *= 1 - max_mask1[:, 0]
            # p_clip = torch.where(p_copy > split_th, p_copy, torch.zeros_like(p_copy))[:, None]
            max_mask2 = torch.where(
                p_copy > self._split_th,
                torch.ones_like(p_copy),
                torch.zeros_like(p_copy),
            )[:, None]
            p_ps2 = max_mask2 * conv

            # this is our final clustered probablity which we then threshold
            # (normally > 0.7) to get our final discrete locations
            p_ps = self._p_aggregation(p_ps1, p_ps2)
            assert p_ps.size(1) == 1

            return p_ps.squeeze(1)

    @classmethod
    def _set_p_aggregation(cls, p_aggr: Union[str, Callable]) -> Callable:
        """
        Returns the p_aggregation callable

        Args:
            p_aggr: probability aggregation
        """

        if isinstance(p_aggr, str):

            if p_aggr == "sum":
                return torch.add
            elif p_aggr == "max":
                return torch.max
            elif p_aggr == "norm_sum":
                return _norm_sum
            else:
                raise ValueError

        else:
            return p_aggr
