import torch
from torch import nn as nn

from deepsmlm.neuralfitter.models.unet_param import UNet2dGN, UNet2d


class DoubleMUnet(nn.Module):
    def __init__(self, ch_in, ch_out, ext_features=0, depth=3, initial_features=64, inter_features=64, \
                                                                                                activation=nn.ReLU(),
                 use_last_nl=True, use_gn=True):
        super().__init__()
        if use_gn:
            self.unet_shared = UNet2dGN(1 + ext_features, inter_features, depth=depth, pad_convs=True,
                                        initial_features=initial_features,
                                        activation=activation)
            self.unet_union = UNet2dGN(ch_in * inter_features, inter_features, depth=depth,
                                       pad_convs=True,
                                       initial_features=initial_features,
                                       activation=activation)
        else:
            self.unet_shared = UNet2d(1 + ext_features, inter_features, depth=depth, pad_convs=True,
                                        initial_features=initial_features,
                                        activation=activation)
            self.unet_union = UNet2d(ch_in * inter_features, inter_features, depth=depth, pad_convs=True,
                                       initial_features=initial_features,
                                       activation=activation)

        assert ch_out in (5, 6)
        self.ch_out = ch_out
        self.mt_heads = nn.ModuleList([MLTHeads(inter_features, group_norm=use_gn) for _ in range(self.ch_out)])

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
            use_gn=param['HyperParameter']['arch_param']['group_normalisation']
        )

    def rescale_last_layer_grad(self, loss, optimizer):
        """

        :param loss: non-reduced loss of size N x C x H x W
        :param optimizer:
        :return: weight, channelwise loss, channelwise weighted loss
        """
        """
        Reduce NCHW channel wise. Division over numel and multiply by ch_out is not needed inside this method, but if
        you want to use loss_wch, or loss_ch directly the numbers would be off by a factor
        """
        loss_ch = loss.sum(-1).sum(-1).sum(0) / loss.numel() * self.ch_out
        head_grads = torch.zeros((self.ch_out, )).to(loss.device)
        weighting = torch.ones_like(head_grads).to(loss.device)

        for i in range(self.ch_out):
            head_grads[i] = torch.autograd.grad(loss_ch[i], self.mt_heads[i].out_conv.weight, retain_graph=True)[
                0].abs().sum()
        optimizer.zero_grad()
        N = (1 / head_grads).sum()
        weighting = weighting / head_grads
        weighting = weighting / N

        loss_wch = (loss_ch * weighting).sum()
        weight_cX_h1_w1 = weighting.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)  # weight tensor of size 1 x C x 1 x 1

        return weight_cX_h1_w1, loss_ch, loss_wch

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

        if not self.training:
            p = self.p_nl(p)

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

    def forward(self, x, external=None):
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
        o = self.unet_union.forward(o)

        o_head = []
        for i in range(self.ch_out):
            o_head.append(self.mt_heads[i].forward(o))
        o = torch.cat(o_head, 1)

        """Apply the final non-linearities"""
        if self._use_last_nl:
            o = self.apply_nonlin(o)

        return o


class MLTHeads(nn.Module):
    def __init__(self, in_channels, activation=nn.ReLU(), last_kernel=1, group_norm=True, padding=True):
        super().__init__()
        groups_1 = min(in_channels, 32)
        groups_2 = min(1, 32)
        padding = True

        self.core = self._make_core(in_channels, groups_1, groups_2, activation, padding, group_norm)
        self.out_conv = nn.Conv2d(in_channels, 1, kernel_size=last_kernel, padding=False)

    def forward(self, x):
        o = self.core.forward(x)
        o = self.out_conv.forward(o)

        return o

    @staticmethod
    def _make_core(in_channels, groups_1, groups_2, activation, padding, group_norm):
        if group_norm:
            return nn.Sequential(nn.GroupNorm(groups_1, in_channels),
                                 nn.Conv2d(in_channels, in_channels,
                                           kernel_size=3, padding=padding),
                                 activation,
                                 nn.GroupNorm(groups_2, in_channels))
        else:
            return nn.Sequential(nn.Conv2d(in_channels, in_channels,
                                           kernel_size=3, padding=padding),
                                 activation)


class DoubleMUNetSeperateBG(DoubleMUnet):
    def __init__(self, ch_in, ch_out, depth=3, initial_features=64, recpt_bg=16, inter_features=64, depth_bg=2,
                 initial_features_bg=16, activation=nn.ReLU(), use_last_nl=True, use_gn=True):
        super().__init__(ch_in=ch_in, ch_out=5, ext_features=1, depth=depth, initial_features=initial_features,
                         inter_features=inter_features, activation=activation,
                         use_last_nl=use_last_nl, use_gn=use_gn)

        if use_gn:
            self.bg_net = UNet2dGN(ch_in, 1, depth=depth_bg, pad_convs=True, initial_features=initial_features_bg,
                                        activation=activation)

        else:
            self.bg_net = UNet2d(ch_in, 1, depth=depth, pad_convs=True,
                                        initial_features=initial_features,
                                        activation=activation)

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
            use_gn=param['HyperParameter']['arch_param']['group_normalisation']
        )

    def forward(self, x):
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

        out = super().forward(x, bg_out.detach())
        return torch.cat((out, bg_out), 1)


class BGNet(nn.Module):
    def __init__(self, ch_in, ch_out, depth=3, initial_features=64, recpt_bg=16, inter_features=64, depth_bg=2,
                 initial_features_bg=16, activation=nn.ReLU(), use_last_nl=True, use_gn=True):
        super().__init__()
        self.ch_out = ch_out  # pseudo channels for easier trainig
        if use_gn:
            self.net = UNet2dGN(in_channels=ch_in,
                                out_channels=1,
                                depth=depth_bg,
                                initial_features=initial_features_bg,
                                pad_convs=True,
                                activation=activation)
        else:
            self.net = UNet2d(in_channels=ch_in,
                                out_channels=1,
                                depth=depth_bg,
                                initial_features=initial_features_bg,
                                pad_convs=True,
                                activation=activation)

    @staticmethod
    def parse(param):
        activation = eval(param['HyperParameter']['arch_param']['activation'])
        return BGNet(
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
            use_gn=param['HyperParameter']['arch_param']['group_normalisation']
        )

    def apply_pnl(self, x):
        return x

    def forward(self, x):
        o = self.net.forward(x)
        if self.ch_out >= 2:
            o = torch.cat((torch.zeros((o.size(0), self.ch_out -1, o.size(-2), o.size(-1))).to(o.device), o), 1)

        return o


if __name__ == '__main__':

    model = DoubleMUNetSeperateBG(3, 6, depth=2, depth_bg=2, initial_features_bg=32, recpt_bg=16, use_last_nl=False)
    x = torch.rand((10, 3, 32, 32))
    y = torch.rand((10, 6, 32, 32))
    optimiser = torch.optim.Adam(model.parameters(), lr=0.0001)
    criterion = torch.nn.MSELoss(reduction='none')
    model.train()
    out = model.forward(x)
    loss = criterion(out, y)
    optimiser.zero_grad()

    print('Done')