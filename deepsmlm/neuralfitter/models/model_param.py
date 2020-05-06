import torch
from torch import nn as nn

from . import unet_param
from ..utils import last_layer_dynamics as lyd


class SimpleSMLMNet(unet_param.UNet2d):

    def __init__(self, ch_in, ch_out, depth=3, initial_features=64, inter_features=64, p_dropout=0.,
                 activation=nn.ReLU(), use_last_nl=True, norm=None, norm_groups=None, norm_head=None,
                 norm_head_groups=None, pool_mode='StrideConv', skip_gn_level=None):
        super().__init__(in_channels=ch_in,
                         out_channels=inter_features,
                         depth=depth,
                         initial_features=initial_features,
                         pad_convs=True,
                         norm=norm,
                         norm_groups=norm_groups,
                         p_dropout=p_dropout,
                         pool_mode=pool_mode,
                         activation=activation,
                         skip_gn_level=skip_gn_level)

        assert ch_out in (5, 6)
        self.ch_out = ch_out
        self.mt_heads = nn.ModuleList(
            [MLTHeads(inter_features, norm=norm_head, norm_groups=norm_head_groups) for _ in range(self.ch_out)])

        self._use_last_nl = use_last_nl

        self.p_nl = torch.sigmoid  # only in inference, during training
        self.phot_nl = torch.sigmoid
        self.xyz_nl = torch.tanh
        self.bg_nl = torch.sigmoid

    @staticmethod
    def parse(param):
        activation = eval(param.HyperParameter.arch_param.activation)
        return SimpleSMLMNet(
            ch_in=param.HyperParameter.channels_in,
            ch_out=param.HyperParameter.channels_out,
            depth=param.HyperParameter.arch_param.depth,
            initial_features=param.HyperParameter.arch_param.initial_features,
            inter_features=param.HyperParameter.arch_param.inter_features,
            p_dropout=param.HyperParameter.arch_param.p_dropout,
            pool_mode=param.HyperParameter.arch_param.pool_mode,
            activation=activation,
            use_last_nl=param.HyperParameter.arch_param.use_last_nl,
            norm=param.HyperParameter.arch_param.norm,
            norm_groups=param.HyperParameter.arch_param.norm_groups,
            norm_head=param.HyperParameter.arch_param.norm_head,
            norm_head_groups=param.HyperParameter.arch_param.norm_head_groups,
            skip_gn_level=param.HyperParameter.arch_param.skip_gn_level
        )

    @staticmethod
    def check_target(y_tar):

        assert y_tar.dim() == 4, "Wrong dim."
        assert y_tar.size(1) == 6, "Wrong num. of channels"
        assert ((y_tar[:, 0] >= 0.) * (y_tar[:, 0] <= 1.)).all(), "Probability outside of the range."
        assert ((y_tar[:, 1] >= 0.) * (y_tar[:, 1] <= 1.)).all(), "Photons outside of the range."
        assert ((y_tar[:, 2:5] >= -1.) * (y_tar[:, 2:5] <= 1.)).all(), "XYZ outside of the range."
        assert ((y_tar[:, 1] >= 0.) * (y_tar[:, 1] <= 1.)).all(), "BG outside of the range."

    def rescale_last_layer_grad(self, loss, optimizer):
        """

        :param loss: non-reduced loss of size N x C x H x W
        :param optimizer:
        :return: weight, channelwise loss, channelwise weighted loss
        """
        return lyd.rescale_last_layer_grad(self.mt_heads, loss, optimizer)

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
        """
        Apply non linearity in all the other channels
        :param o:
        :return:
        """
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


class SMLMNetBG(SimpleSMLMNet):
    def __init__(self, ch_in, ch_out, depth=3, initial_features=64, inter_features=64, p_dropout=0.,
                 activation=nn.ReLU(), use_last_nl=True, norm=None, norm_groups=None, norm_bg=None,
                 norm_bg_groups=None, norm_head=None, norm_head_groups=None, pool_mode='MaxPool', detach_bg=False,
                 skip_gn_level=None):

        super().__init__(ch_in + 1, ch_out - 1, depth, initial_features, inter_features, p_dropout, activation,
                         use_last_nl,
                         norm, norm_groups,
                         pool_mode=pool_mode,
                         skip_gn_level=skip_gn_level)
        assert ch_out == 6
        self.total_ch_out = ch_out
        self.detach_bg = detach_bg

        self.bg_net = deepsmlm.neuralfitter.models.unet_param.UNet2d(1, 1, 2, 48, pad_convs=True, norm=norm_bg,
                                                                     norm_groups=norm_bg_groups,
                                                                     activation=activation)

    @staticmethod
    def parse(param):
        activation = eval(param['HyperParameter']['arch_param']['activation'])
        return SMLMNetBG(
            ch_in=param['HyperParameter']['channels_in'],
            ch_out=param['HyperParameter']['channels_out'],
            depth=param['HyperParameter']['arch_param']['depth'],
            initial_features=param['HyperParameter']['arch_param']['initial_features'],
            inter_features=param['HyperParameter']['arch_param']['inter_features'],
            p_dropout=param['HyperParameter']['arch_param']['p_dropout'],
            pool_mode=param['HyperParameter']['arch_param']['pool_mode'],
            activation=activation,
            use_last_nl=param['HyperParameter']['arch_param']['use_last_nl'],
            norm=param['HyperParameter']['arch_param']['norm'],
            norm_groups=param['HyperParameter']['arch_param']['norm_groups'],
            norm_bg=param['HyperParameter']['arch_param']['norm_bg'],
            norm_bg_groups=param['HyperParameter']['arch_param']['norm_bg_groups'],
            detach_bg=param['HyperParameter']['arch_param']['detach_bg'],
            skip_gn_level=param.HyperParameter.arch_param.skip_gn_level
        )

    def forward(self, x):
        bg = self.bg_net.forward(x)
        if self.detach_bg:
            xbg = torch.cat((x, bg.detach()), 1)
        else:
            xbg = torch.cat((x, bg), 1)

        x = super().forward(xbg)
        return x


class DoubleMUnet(nn.Module):
    def __init__(self, ch_in, ch_out, ext_features=0, depth=3, initial_features=64, inter_features=64,
                 activation=nn.ReLU(), use_last_nl=True, norm=None, norm_groups=None, norm_head=None,
                 norm_head_groups=None, pool_mode='Conv2d', skip_gn_level=None):
        super().__init__()

        self.unet_shared = unet_param.UNet2d(1 + ext_features, inter_features, depth=depth, pad_convs=True,
                                             initial_features=initial_features,
                                             activation=activation, norm=norm, norm_groups=norm_groups,
                                             pool_mode=pool_mode,
                                             skip_gn_level=skip_gn_level)
        self.unet_union = unet_param.UNet2d(ch_in * inter_features, inter_features, depth=depth, pad_convs=True,
                                            initial_features=initial_features,
                                            activation=activation, norm=norm, norm_groups=norm_groups,
                                            pool_mode=pool_mode,
                                            skip_gn_level=skip_gn_level)

        assert ch_in in (1, 3)
        assert ch_out in (5, 6)
        self.ch_in = ch_in
        self.ch_out = ch_out
        self.mt_heads = nn.ModuleList(
            [MLTHeads(inter_features, norm=norm_head, norm_groups=norm_head_groups) for _ in range(self.ch_out)])

        self._use_last_nl = use_last_nl

        self.p_nl = torch.sigmoid  # only in inference, during training
        self.phot_nl = torch.sigmoid
        self.xyz_nl = torch.tanh
        self.bg_nl = torch.sigmoid

    @staticmethod
    def parse(param):
        activation = eval(param['HyperParameter']['arch_param']['activation'])
        return DoubleMUnet(
            ch_in=param['HyperParameter']['channels_in'],
            ch_out=param['HyperParameter']['channels_out'],
            ext_features=0,
            depth=param['HyperParameter']['arch_param']['depth'],
            initial_features=param['HyperParameter']['arch_param']['initial_features'],
            inter_features=param['HyperParameter']['arch_param']['inter_features'],
            activation=activation,
            use_last_nl=param['HyperParameter']['arch_param']['use_last_nl'],
            norm=param['HyperParameter']['arch_param']['norm'],
            norm_groups=param['HyperParameter']['arch_param']['norm_groups'],
            norm_head=param['HyperParameter']['arch_param']['norm_head'],
            norm_head_groups=param['HyperParameter']['arch_param']['norm_head_groups'],
            pool_mode=param['HyperParameter']['arch_param']['pool_mode'],
            skip_gn_level=param.HyperParameter.arch_param.skip_gn_level
        )

    def rescale_last_layer_grad(self, loss, optimizer):
        """

        :param loss: non-reduced loss of size N x C x H x W
        :param optimizer:
        :return: weight, channelwise loss, channelwise weighted loss
        """
        return lyd.rescale_last_layer_grad(self.mt_heads, loss, optimizer)

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
        """
        Apply non linearity in all the other channels
        :param o:
        :return:
        """
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

    def forward(self, x, external=None, force_no_p_nl=False):
        """

        Args:
            x:
            external:
            force_no_p_nl:

        Returns:

        """
        if self.ch_in == 3:
            x0 = x[:, [0]]
            x1 = x[:, [1]]
            x2 = x[:, [2]]
            if external is not None:
                x0 = torch.cat((x0, external), 1)
                x1 = torch.cat((x1, external), 1)
                x2 = torch.cat((x2, external), 1)

            o0 = self.unet_shared.forward(x0)
            o1 = self.unet_shared.forward(x1)
            o2 = self.unet_shared.forward(x2)

            o = torch.cat((o0, o1, o2), 1)

        elif self.ch_in == 1:
            o = self.unet_shared.forward(x)

        o = self.unet_union.forward(o)

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


class MLTHeads(nn.Module):
    def __init__(self, in_channels, activation=nn.ReLU(), last_kernel=1, norm=None, norm_groups=None, padding=True):
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

        self.core = self._make_core(in_channels, groups_1, groups_2, activation, padding, self.norm)
        self.out_conv = nn.Conv2d(in_channels, 1, kernel_size=last_kernel, padding=False)

    def forward(self, x):
        o = self.core.forward(x)
        o = self.out_conv.forward(o)

        return o

    @staticmethod
    def _make_core(in_channels, groups_1, groups_2, activation, padding, norm):
        if norm == 'GroupNorm':
            return nn.Sequential(nn.GroupNorm(groups_1, in_channels),
                                 nn.Conv2d(in_channels, in_channels,
                                           kernel_size=3, padding=padding),
                                 activation,
                                 # nn.GroupNorm(groups_2, in_channels)
                                 )
        elif norm is None:
            return nn.Sequential(nn.Conv2d(in_channels, in_channels,
                                           kernel_size=3, padding=padding),
                                 activation)
        else:
            raise NotImplementedError


class DoubleMUNetSeperateBG(SimpleSMLMNet):
    def __init__(self, ch_in, ch_out, depth=3, initial_features=64, recpt_bg=16, inter_features=64, depth_bg=2,
                 initial_features_bg=16, activation=nn.ReLU(), use_last_nl=True, norm=None, norm_groups=None,
                 norm_bg=None, norm_bg_groups=None, pool_mode='Conv2d', skip_gn_level=None):
        super().__init__(ch_in=ch_in, ch_out=5, depth=depth, initial_features=initial_features,
                         inter_features=inter_features, activation=activation,
                         use_last_nl=use_last_nl, norm=norm, norm_groups=norm_groups, pool_mode=pool_mode,
                         skip_gn_level=skip_gn_level)

        self.bg_net = unet_param.UNet2d(ch_in, 1, depth=depth_bg, pad_convs=True,
                                        initial_features=initial_features_bg,
                                        activation=activation, norm=norm_bg,
                                        norm_groups=norm_bg_groups, pool_mode=pool_mode,
                                        skip_gn_level=skip_gn_level)

        self.bg_nl = torch.tanh
        self.bg_recpt = recpt_bg

    @staticmethod
    def parse(param):
        activation = eval(param['HyperParameter']['arch_param']['activation'])
        return DoubleMUNetSeperateBG(
            ch_in=param['HyperParameter']['channels_in'],
            ch_out=param['HyperParameter']['channels_out'],
            depth=param['HyperParameter']['arch_param']['depth'],
            initial_features=param['HyperParameter']['arch_param']['initial_features'],
            recpt_bg=param['HyperParameter']['arch_param']['recpt_bg'],
            depth_bg=param['HyperParameter']['arch_param']['depth_bg'],
            initial_features_bg=param['HyperParameter']['arch_param']['initial_features_bg'],
            inter_features=param['HyperParameter']['arch_param']['inter_features'],
            activation=activation,
            use_last_nl=param['HyperParameter']['arch_param']['use_last_nl'],
            norm=param['HyperParameter']['arch_param']['norm'],
            norm_groups=param['HyperParameter']['arch_param']['norm_groups'],
            norm_bg=param['HyperParameter']['arch_param']['norm_bg'],
            norm_bg_groups=param['HyperParameter']['arch_param']['norm_bg_groups'],
            pool_mode=param['HyperParameter']['arch_param']['pool_mode'],
            skip_gn_level=param.HyperParameter.arch_param.skip_gn_level
        )

    def forward(self, x, force_no_p_nl=False):
        """

        :param x:
        :return:
        """
        """
        During training, limit the 
        """
        if self.training:
            bg_out = torch.zeros_like(x[:, [0]])
            assert x.size(-1) % self.bg_recpt == 0
            assert x.size(-2) % self.bg_recpt == 0
            n_x = x.size(-2) // self.bg_recpt
            n_y = x.size(-1) // self.bg_recpt
            for i in range(n_x):
                for j in range(n_y):
                    ii = slice(i * self.bg_recpt, (i + 1) * self.bg_recpt)
                    jj = slice(j * self.bg_recpt, (j + 1) * self.bg_recpt)
                    bg_out[:, :, ii, jj] = self.bg_net.forward(x[:, :, ii, jj])
        else:
            bg_out = self.bg_net.forward(x)

        out = super().forward(x, bg_out.detach(), force_no_p_nl=force_no_p_nl)
        return torch.cat((out, bg_out), 1)


class BGNet(nn.Module):
    def __init__(self, ch_in, ch_out, depth_bg=2, initial_features_bg=16, recpt_field=None, activation=nn.ReLU(),
                 norm=None, norm_groups=None, pool_mode='MaxPool', skip_gn_level=None):
        super().__init__()
        self.ch_out = ch_out  # pseudo channels for easier trainig
        self.recpt_field = recpt_field
        self.net = unet_param.UNet2d(in_channels=ch_in,
                                     out_channels=1,
                                     depth=depth_bg,
                                     initial_features=initial_features_bg,
                                     pad_convs=True,
                                     activation=activation,
                                     norm=norm,
                                     norm_groups=norm_groups,
                                     pool_mode=pool_mode,
                                     skip_gn_level=skip_gn_level)

    @staticmethod
    def parse(param):
        activation = eval(param['HyperParameter']['arch_param']['activation'])
        return BGNet(
            ch_in=param.HyperParameter.channels_in,
            ch_out=param.HyperParameter.channels_out,
            depth_bg=param.HyperParameter.arch_param.depth_bg,
            initial_features_bg=param.HyperParameter.arch_param.initial_features_bg,
            recpt_field=param.HyperParameter.arch_param.recpt_bg,
            activation=activation,
            norm=param.HyperParameter.arch_param.norm_bg,
            norm_groups=param.HyperParameter.arch_param.norm_bg_groups,
            pool_mode=param.HyperParameter.arch_param.pool_mode,
            skip_gn_level=param.HyperParameter.arch_param.skip_gn_level
        )

    @staticmethod
    def apply_pnl(x):
        """
        Dummy method.
        :param x:
        :return:
        """
        return x

    def forward_recpt(self, x):
        """
        Forward data in limited receptieve field.
        Data dimensions must be multiples of recpt field.
        :param x:
        :return:
        """
        hs = x.size(-2)
        ws = x.size(-1)

        assert hs % self.recpt_field == 0
        assert ws % self.recpt_field == 0

        n_x, n_y = hs // self.recpt_field, ws // self.recpt_field
        o = torch.zeros_like(x[:, [0]])
        for i in range(n_x):
            ii = slice(i * self.recpt_field, (i + 1) * self.recpt_field)
            for j in range(n_y):
                jj = slice(j * self.recpt_field, (j + 1) * self.recpt_field)
                o[:, :, ii, jj] = self.net.forward(x[:, :, ii, jj])
        return o

    def forward(self, x):
        if self.recpt_field is not None and self.training:
            o = self.forward_recpt(x)
        else:
            o = self.net.forward(x)

        if self.ch_out >= 2:
            o = torch.cat((torch.zeros((o.size(0), self.ch_out - 1, o.size(-2), o.size(-1))).to(o.device), o), 1)

        return o
