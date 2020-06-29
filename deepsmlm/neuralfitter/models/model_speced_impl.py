from typing import Union

import torch
from torch import nn

from . import model_param


class SigmaMUNet(model_param.DoubleMUnet):
    ch_out = 10
    out_channels_heads = (1, 4, 4, 1)  # p head, phot,xyz_mu head, phot,xyz_sig head, bg head

    sigmoid_ch_ix = [0, 1, 5, 6, 7, 8, 9]  # channel indices with respective activation function
    tanh_ch_ix = [2, 3, 4]

    p_ch_ix = [0]  # channel indices of the respective parameters
    pxyz_mu_ch_ix = slice(1, 5)
    pxyz_sig_ch_ix = slice(5, 9)
    bg_ch_ix = [10]
    sigma_eps = 0.01

    def __init__(self, ch_in: int, *, depth_shared: int, depth_union: int, initial_features: int, inter_features: int,
                 norm=None, norm_groups=None, norm_head=None, norm_head_groups=None, pool_mode='StrideConv',
                 skip_gn_level: Union[None, bool] = None, activation=nn.ReLU()):
        super().__init__(ch_in=ch_in, ch_out=self.ch_out, depth_shared=depth_shared, depth_union=depth_union,
                         initial_features=initial_features, inter_features=inter_features,
                         norm=norm, norm_groups=norm_groups, norm_head=norm_head,
                         norm_head_groups=norm_head_groups, pool_mode=pool_mode,
                         skip_gn_level=skip_gn_level, activation=activation,
                         use_last_nl=False)

        self.mt_heads = torch.nn.ModuleList(
            [model_param.MLTHeads(in_channels=inter_features, out_channels=ch_out,
                                  norm=norm_head, norm_groups=norm_head_groups)
             for ch_out in self.out_channels_heads]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._forward_core(external=None, x=x)

        """Forward through the respective heads"""
        x_heads = [mt_head.forward(x) for mt_head in self.mt_heads]
        x = torch.cat(x_heads, dim=1)

        """Apply non linearities"""
        x[:, self.sigmoid_ch_ix] = torch.sigmoid(x[:, self.sigmoid_ch_ix])
        x[:, self.tanh_ch_ix] = torch.tanh(x[:, self.tanh_ch_ix])

        # add epsilon to sigmas and rescale
        x[:, self.pxyz_sig_ch_ix] = x[:, self.pxyz_sig_ch_ix] * 3 + self.sigma_eps

        return x

    def apply_detection_nonlin(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def apply_nonlin(self, o: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @classmethod
    def parse(cls, param, **kwargs):
        activation = eval(param.HyperParameter.arch_param.activation)
        return cls(
            ch_in=param.HyperParameter.channels_in,
            depth_shared=param.HyperParameter.arch_param.depth_shared,
            depth_union=param.HyperParameter.arch_param.depth_union,
            initial_features=param.HyperParameter.arch_param.initial_features,
            inter_features=param.HyperParameter.arch_param.inter_features,
            activation=activation,
            norm=param.HyperParameter.arch_param.norm,
            norm_groups=param.HyperParameter.arch_param.norm_groups,
            norm_head=param.HyperParameter.arch_param.norm_head,
            norm_head_groups=param.HyperParameter.arch_param.norm_head_groups,
            pool_mode=param.HyperParameter.arch_param.pool_mode,
            skip_gn_level=param.HyperParameter.arch_param.skip_gn_level
        )
