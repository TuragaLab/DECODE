from typing import Callable, Sequence, Union, Optional

import torch
from torch import nn

from . import model_param
from .. import spec


class SigmaMUNet(model_param.DoubleMUnet):
    # pxyz_mu_ch_ix = slice(1, 5)
    # pxyz_sig_ch_ix = slice(5, 9)
    # bg_ch_ix = [10]
    sigma_eps_default = 0.001

    def __init__(
        self,
        *,
        ch_in_map: Sequence[Sequence[int]],
        ch_out_heads: Sequence[int] = (1, 4, 4, 1),
        ch_map: spec.ModelChannelMapGMM,
        depth_shared: int,
        depth_union: int,
        initial_features: int,
        inter_features: int,
        norm=None,
        norm_groups=None,
        norm_head=None,
        norm_head_groups=None,
        pool_mode="StrideConv",
        upsample_mode="bilinear",
        skip_gn_level: Union[None, bool] = None,
        activation=nn.ReLU(),
        activation_last: Optional[dict[Union[str, Callable], list[int]]] = None,
        disabled_attributes=None,
        kaiming_normal=True
    ):
        """

        Args:
            ch_in_map: list of list of input channels for each shared net
            ch_out_heads: defaults to [1, 4, 4, 1] as prob, photxyz_mu, photxyz_sig, bg
            depth_shared:
            depth_union:
            initial_features:
            inter_features:
            norm:
            norm_groups:
            norm_head:
            norm_head_groups:
            pool_mode:
            upsample_mode:
            skip_gn_level:
            activation:
            disabled_attributes:
            kaiming_normal:
        """

        super().__init__(ch_in_map=ch_in_map, ch_out=sum(ch_out_heads), depth_shared=depth_shared, depth_union=depth_union,
                         initial_features=initial_features, inter_features=inter_features, activation=activation,
                         use_last_nl=False, norm=norm, norm_groups=norm_groups, norm_head=norm_head,
                         norm_head_groups=norm_head_groups, pool_mode=pool_mode, upsample_mode=upsample_mode,
                         skip_gn_level=skip_gn_level, disabled_attributes=disabled_attributes)

        self._ch_map = ch_map
        self._ix_sigmoid = self._ch_map.ix_prob \
                           + self._ch_map.ix_phot \
                           + self._ch_map.ix_sig \
                           + self._ch_map.ix_bg
        self._ix_tanh = self._ch_map.ix_xyz
        self.mt_heads = torch.nn.ModuleList(
            [
                model_param.MLTHeads(
                    in_channels=inter_features,
                    out_channels=ch_out,
                    activation=activation,
                    last_kernel=1,
                    padding=1,
                    norm=norm_head,
                    norm_groups=norm_head_groups,
                )
                for ch_out in ch_out_heads
            ]
        )

        # register sigma as parameter such that it is stored in the models state dict
        # and loaded correctly
        self.register_parameter(
            "sigma_eps",
            torch.nn.Parameter(
                torch.tensor([self.sigma_eps_default]), requires_grad=False
            ),
        )

        if kaiming_normal:
            self.apply(self.weight_init)

            # custom
            torch.nn.init.kaiming_normal_(
                self.mt_heads[0].core[0].weight, mode="fan_in", nonlinearity="relu"
            )
            torch.nn.init.kaiming_normal_(
                self.mt_heads[0].out_conv.weight, mode="fan_in", nonlinearity="linear"
            )
            torch.nn.init.constant_(self.mt_heads[0].out_conv.bias, -6.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._forward_core(x)

        # forward through the respective heads
        x_heads = [mt_head.forward(x) for mt_head in self.mt_heads]
        x = torch.cat(x_heads, dim=1)

        # clamp prob before sigmoid
        x[:, self._ch_map.ix_prob] = torch.clamp(
            x[:, self._ch_map.ix_prob], min=-8.0, max=8.0
        )

        # apply non linearities
        x[:, self._ix_sigmoid] = torch.sigmoid(x[:, self._ix_sigmoid])
        x[:, self._ix_tanh] = torch.tanh(x[:, self._ix_tanh])

        # add epsilon to sigmas and rescale
        x[:, self._ch_map.ix_sig] = x[:, self._ch_map.ix_sig] * 3 + self.sigma_eps

        # disabled attributes get set to constants
        if self.disabled_attr_ix is not None:
            raise NotImplementedError(f"This needs to be adapted to multi-code model.")
            for ix in self.disabled_attr_ix:
                # Set means to 0
                x[:, 1 + ix] = x[:, 1 + ix] * 0
                # Set sigmas to 0.1
                x[:, 5 + ix] = x[:, 5 + ix] * 0 + 0.1

        return x

    def apply_detection_nonlin(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def apply_nonlin(self, o: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @staticmethod
    def weight_init(m):
        """
        Apply Kaiming normal init. Call this recursively by model.apply(model.weight_init)

        Args:
            m: model

        """
        if isinstance(m, torch.nn.Conv2d):
            torch.nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
