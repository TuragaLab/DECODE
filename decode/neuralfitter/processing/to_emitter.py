from abc import ABC, abstractmethod
from typing import Any, Callable, Optional, Union

import torch

from ...emitter import emitter
from .. import spec


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
        ch_map: spec.ModelChannelMap,
        xy_unit: str,
        px_size: Optional[tuple[float, float]] = None,
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
        self._ch_map = ch_map
        self._device = device

    def forward(self, x: torch.Tensor) -> emitter.EmitterSet:
        """
        Forward model output tensor through post-processing and return EmitterSet.
        Will include sigma values in  EmitterSet if mapping was provided initially.

        Args:
            x: tensor from which values are extracted
        """
        if x.dim() != 4:
            raise ValueError("Input tensor must be 4D (N, C, H, W)")

        x = x.clone().detach()
        prob = x[:, self._ch_map.ix_prob]
        mask = self._mask(prob)

        # aggregate probabilities by max for features that are irrespective of code
        mask_agg = mask.max(dim=1)[0]

        frame_ix, features = self._lookup_features(x, mask_agg)
        features = self._ch_map.split_tensor(features.permute(1, 0))  # expects N x C
        features = {k: v.squeeze(-1) for k, v in features.items()}
        features["frame_ix"] = frame_ix
        features["prob"] = features["prob"].max(-1)[0]

        # ToDo: Change if code comes not from probabilities but from multiple phot
        # ToDo: channels
        # if hasattr(self._ch_map, "ix_code") and self._ch_map.ix_code is not None:
        features["code"] = self._look_up_code(mask)

        features = {k: v.to(self._device) for k, v in features.items()}

        return emitter.EmitterSet(
            **features,
            xy_unit=self._xy_unit,
            px_size=self._px_size,
        )

    def _mask(self, p: torch.Tensor) -> torch.BoolTensor:
        # mask by callable or threshold
        if not callable(self._mask_impl):
            return p >= self._mask_impl
        return self._mask_impl(p)

    @staticmethod
    def _lookup_features(
        features: torch.Tensor, active_px: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Extract features from tensor for active pixels.

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

    def _look_up_code(self, m: torch.BoolTensor) -> torch.Tensor:
        """
        Extract code from tensor for active pixels.

        Args:
            m: size :math:`(N, C, H, W)`

        Returns:
            torch.Tensor: code, size :math: `M`
        """
        if m.dim() != 4:
            raise ValueError(
                f"Mask must be 4D (N, C, H, W), " f"got tensor of size {m.size()}"
            )

        m_agg = m.max(dim=1)[0]

        code = torch.arange(m.size(1), device=m.device)
        code = code.view(1, -1, 1, 1).repeat(m.size(0), 1, m.size(2), m.size(3))

        code = (code * m).max(1, keepdim=True)[0]

        # this must be in line with _lookup_features
        code = code.permute(1, 0, 2, 3)[:, m_agg].squeeze(0)
        return code


def _norm_sum(*args):
    return torch.clamp(torch.add(*args), 0.0, 1.0)


class ToEmitterSpatialIntegration(ToEmitterLookUpPixelwise):
    def __init__(
        self,
        raw_th: float,
        ch_map: spec.ModelChannelMap,
        xy_unit: str,
        px_size: tuple[float, float] = None,
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
            ch_map=ch_map,
        )
        self._raw_th = raw_th
        self._split_th = _split_th
        self._p_aggregation = self._set_p_aggregation(p_aggregation)

    def forward(self, x: torch.Tensor) -> emitter.EmitterSet:
        x[:, 0] = self._non_max_suppression(x[:, 0])
        return super().forward(x)

    def _non_max_suppression(self, p: torch.Tensor) -> torch.Tensor:
        """
        Non-maximum suppression-like algorithm to handle local clusters and
        to distinguish between local clusters that are a single emitter and
        those that are multiple emitters

        Args:
            p: probability map of size :math:`(N, C, H, W)`
        """
        with torch.no_grad():
            p_copy = p.clone()

            # Probability values > 0.3 are regarded as possible locations
            p_clip = torch.where(p > self._raw_th, p, torch.zeros_like(p))

            # localize maximum values within a 3x3 patch
            pool = torch.nn.functional.max_pool2d(p_clip, 3, 1, padding=1)
            max_mask1 = torch.eq(p, pool).float()

            # Add probability values from the 4 adjacent pixels
            diag = 0.0  # 1/np.sqrt(2)
            filt = (
                torch.tensor(
                    [[diag, 1.0, diag], [1, 1, 1], [diag, 1, diag]],
                    device=p.device
                )
                .unsqueeze(0)
                .unsqueeze(0)
            )
            conv = torch.cat([torch.nn.functional.conv2d(pp.unsqueeze(1), filt, padding=1) for pp in torch.unbind(p, dim=1)], dim=1)
            p_ps1 = max_mask1 * conv

            # to identify two fluorophores in adjacent pixels we look
            # for probablity values > 0.6 that are not part of the first mask
            p_copy *= 1 - max_mask1
            max_mask2 = torch.where(
                p_copy > self._split_th,
                torch.ones_like(p_copy),
                torch.zeros_like(p_copy),
            )
            p_ps2 = max_mask2 * conv

            # this is our final clustered probability which we then threshold
            # (normally > 0.7) to get our final discrete locations
            p_ps = self._p_aggregation(p_ps1, p_ps2)
            return p_ps

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
                raise NotImplementedError(f"Unknown p_aggregation {p_aggr}")

        else:
            return p_aggr
