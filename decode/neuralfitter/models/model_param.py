import torch
from torch import nn as nn

from . import unet_param
from ..utils import last_layer_dynamics as lyd


class SimpleSMLMNet(unet_param.UNet2d):
    def __init__(
        self,
        ch_in,
        ch_out,
        depth=3,
        initial_features=64,
        inter_features=64,
        p_dropout=0.0,
        activation=nn.ReLU(),
        use_last_nl=True,
        norm=None,
        norm_groups=None,
        norm_head=None,
        norm_head_groups=None,
        pool_mode="StrideConv",
        upsample_mode="bilinear",
        skip_gn_level=None,
    ):
        super().__init__(
            in_channels=ch_in,
            out_channels=inter_features,
            depth=depth,
            initial_features=initial_features,
            pad_convs=True,
            norm=norm,
            norm_groups=norm_groups,
            p_dropout=p_dropout,
            pool_mode=pool_mode,
            activation=activation,
            skip_gn_level=skip_gn_level,
        )

        assert ch_out in (5, 6)
        self.ch_out = ch_out
        self.mt_heads = nn.ModuleList(
            [
                MLTHeads(
                    inter_features,
                    norm=norm_head,
                    norm_groups=norm_head_groups,
                    padding=1,
                    activation=activation,
                )
                for _ in range(self.ch_out)
            ]
        )

        self._use_last_nl = use_last_nl

        self.p_nl = torch.sigmoid  # only in inference, during training
        self.phot_nl = torch.sigmoid
        self.xyz_nl = torch.tanh
        self.bg_nl = torch.sigmoid

    @staticmethod
    def check_target(y_tar):

        assert y_tar.dim() == 4, "Wrong dim."
        assert y_tar.size(1) == 6, "Wrong num. of channels"
        assert (
            (y_tar[:, 0] >= 0.0) * (y_tar[:, 0] <= 1.0)
        ).all(), "Probability outside of the range."
        assert (
            (y_tar[:, 1] >= 0.0) * (y_tar[:, 1] <= 1.0)
        ).all(), "Photons outside of the range."
        assert (
            (y_tar[:, 2:5] >= -1.0) * (y_tar[:, 2:5] <= 1.0)
        ).all(), "XYZ outside of the range."
        assert (
            (y_tar[:, 1] >= 0.0) * (y_tar[:, 1] <= 1.0)
        ).all(), "BG outside of the range."

    def rescale_last_layer_grad(self, loss, optimizer):
        """

        :param loss: non-reduced loss of size N x C x H x W
        :param optimizer:
        :return: weight, channelwise loss, channelwise weighted loss
        """
        return lyd.weight_by_gradient(self.mt_heads, loss, optimizer)

    def apply_pnl(self, o):
        """
        Apply nonlinearity (sigmoid) to p channel. This is combined during training in the loss function.
        Only use when not training
        :param o:
        :return:
        """
        o[:, [0]] = self.p_nl(o[:, [0]])
        return o

    def apply_nonlin(self, o):
        # apply non linearity in all the other channels

        # phot, xyz
        p = o[:, [0]]  # leave unused
        phot = o[:, [1]]
        xyz = o[:, 2:5]

        phot = self.phot_nl(phot)
        xyz = self.xyz_nl(xyz)

        if self.ch_out == 5:
            o = torch.cat((p, phot, xyz), 1)
            return o
        elif self.ch_out == 6:
            bg = o[:, [5]]
            bg = self.bg_nl(bg)

            o = torch.cat((p, phot, xyz, bg), 1)
            return o

    def forward(self, x, force_no_p_nl=False):
        o = super().forward(x)

        o_head = []
        for i in range(self.ch_out):
            o_head.append(self.mt_heads[i].forward(o))
        o = torch.cat(o_head, 1)

        """Apply the final non-linearities"""
        if not self.training and not force_no_p_nl:
            o[:, [0]] = self.p_nl(o[:, [0]])

        if self._use_last_nl:
            o = self.apply_nonlin(o)

        return o


class DoubleMUnet(nn.Module):
    p_nl = torch.sigmoid  # only in inference, during training
    phot_nl = torch.sigmoid
    xyz_nl = torch.tanh
    bg_nl = torch.sigmoid

    def __init__(
        self,
        ch_in_map: list[list[int]],
        ch_out: int,
        depth_shared: int = 3,
        depth_union: int = 3,
        initial_features: int = 64,
        inter_features: int = 64,
        activation=nn.ReLU(),
        use_last_nl=True,
        norm=None,
        norm_groups=None,
        norm_head=None,
        norm_head_groups=None,
        pool_mode="StrideConv",
        upsample_mode="bilinear",
        skip_gn_level=None,
        disabled_attributes=None,
    ):
        super().__init__()

        if len({len(m) for m in ch_in_map}) != 1:
            raise ValueError("All maps must have the same number of channels.")
        n_groups = len(ch_in_map)
        n_ch_group = len(ch_in_map[0])

        self.ch_in_map = ch_in_map
        self.ch_out = ch_out
        self._n_groups = n_groups
        self._n_ch_group = n_ch_group
        self._use_last_nl = use_last_nl

        self.unet_shared = unet_param.UNet2d(
            n_ch_group,
            inter_features,
            depth=depth_shared,
            pad_convs=True,
            initial_features=initial_features,
            activation=activation,
            norm=norm,
            norm_groups=norm_groups,
            pool_mode=pool_mode,
            upsample_mode=upsample_mode,
            skip_gn_level=skip_gn_level,
        )

        self.unet_union = unet_param.UNet2d(
            n_groups * inter_features,
            inter_features,
            depth=depth_union,
            pad_convs=True,
            initial_features=initial_features,
            activation=activation,
            norm=norm,
            norm_groups=norm_groups,
            pool_mode=pool_mode,
            upsample_mode=upsample_mode,
            skip_gn_level=skip_gn_level,
        )

        self.mt_heads = nn.ModuleList(
            [
                MLTHeads(
                    inter_features,
                    out_channels=1,
                    last_kernel=1,
                    norm=norm_head,
                    norm_groups=norm_head_groups,
                    padding=1,
                    activation=activation,
                )
                for _ in range(self.ch_out)
            ]
        )

        # convert to list
        if disabled_attributes is None or isinstance(
            disabled_attributes, (tuple, list)
        ):
            self.disabled_attr_ix = disabled_attributes
        else:
            self.disabled_attr_ix = [disabled_attributes]

    def rescale_last_layer_grad(self, loss, optimizer):
        """
        Rescales the weight as by the last layer's gradient

        Args:
            loss:
            optimizer:

        Returns:
            weight, channelwise loss, channelwise weighted loss

        """
        return lyd.weight_by_gradient(self.mt_heads, loss, optimizer)

    def apply_detection_nonlin(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply detection non-linearity. Useful for non-training situations. When BCEWithLogits loss is used, do not use this
         during training (because it's already included in the loss).

        Args:
            o: model output

        """
        x[:, [0]] = self.p_nl(x[:, [0]])
        return x

    def apply_nonlin(self, o: torch.Tensor) -> torch.Tensor:
        """
        Apply non-linearity to all but the detection channel.

        Args:
            o:

        """
        raise NotImplementedError("Only implemented for single channel output")
        # Apply for phot, xyz
        p = o[:, [0]]  # leave unused
        phot = o[:, [1]]
        xyz = o[:, 2:5]

        phot = self.phot_nl(phot)
        xyz = self.xyz_nl(xyz)

        if self.ch_out == 5:
            o = torch.cat((p, phot, xyz), 1)
            return o
        elif self.ch_out == 6:
            bg = o[:, [5]]
            bg = self.bg_nl(bg)

            o = torch.cat((p, phot, xyz, bg), 1)
            return o

    def forward(self, x, force_no_p_nl=False):
        """

        Args:
            x:
            force_no_p_nl:

        Returns:

        """
        o = self._forward_core(x)

        o_head = []
        for i in range(self.ch_out):
            o_head.append(self.mt_heads[i].forward(o))
        o = torch.cat(o_head, 1)

        # apply the final non-linearities
        if not self.training and not force_no_p_nl:
            o[:, [0]] = self.p_nl(o[:, [0]])

        if self._use_last_nl:
            o = self.apply_nonlin(o)

        return o

    def _forward_core(self, x) -> torch.Tensor:
        # core, i.e. shared and union networks
        out_shared = [None] * self._n_groups

        # map input channels through shared network iteratively
        for i, ch_map in enumerate(self.ch_in_map):
            out_shared[i] = self.unet_shared.forward(x[:, ch_map, :, :])

        out_shared = torch.cat(out_shared, 1)
        out_union = self.unet_union.forward(out_shared)

        return out_union


class MLTHeads(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        last_kernel,
        norm,
        norm_groups,
        padding: int,
        activation,
    ):
        super().__init__()
        self.norm = norm
        self.norm_groups = norm_groups
        if self.norm is not None:
            groups_1 = min(in_channels, self.norm_groups)
            groups_2 = min(1, self.norm_groups)
        else:
            groups_1 = None
            groups_2 = None

        padding = padding

        self.core = self._make_core(
            in_channels, groups_1, groups_2, activation, padding, self.norm
        )
        self.out_conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=last_kernel, padding=0
        )

    def forward(self, x):
        o = self.core.forward(x)
        o = self.out_conv.forward(o)

        return o

    @staticmethod
    def _make_core(in_channels, groups_1, groups_2, activation, padding, norm):
        if norm == "GroupNorm":
            return nn.Sequential(
                nn.GroupNorm(groups_1, in_channels),
                nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=padding),
                activation,
                # nn.GroupNorm(groups_2, in_channels)
            )
        elif norm is None:
            return nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=padding),
                activation,
            )
        else:
            raise NotImplementedError
