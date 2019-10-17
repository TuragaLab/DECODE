import torch

from deepsmlm.neuralfitter.models.unet_model import *


class DoubleOffsetUNet(nn.Module):

    def __init__(self, n_channels, n_classes, n_intermediate):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.n_intermediate = n_intermediate

        self.u_nets = []
        self.u_net0 = UNet(n_channels=n_channels, n_classes=n_intermediate+1)
        self.u_net1 = UNet(n_channels=n_intermediate+n_channels, n_classes=n_classes-1)

        self.p_nl_inference = torch.sigmoid  # identity function since sigmoid is now in loss.
        self.i_nl = torch.sigmoid
        self.xyz_nl = torch.tanh
        self.bg_nl = torch.sigmoid

    # def parameters(self):
    #     return nn.ParameterList([torch.nn.Parameter(self.u_nets[0].parameters()), torch.nn.Parameter(self.u_nets[1].parameters())])

    def apply_pnl(self, output):
        output[:, [0]] = self.p_nl_inference(output[:, [0]])
        return output

    def forward(self, x):
        """

        :param x:
        :return:
        """
        x1 = self.u_net0.inc(x)
        x2 = self.u_net0.down1(x1)
        x3 = self.u_net0.down2(x2)
        x4 = self.u_net0.down3(x3)
        x5 = self.u_net0.down4(x4)
        x6 = self.u_net0.up1(x5, x4)
        x7 = self.u_net0.up2(x6, x3)
        x8 = self.u_net0.up3(x7, x2)
        x9 = self.u_net0.up4(x8, x1)
        x_out_u1 = self.u_net0.outc(x9)

        bg = x_out_u1[:, [0]]

        x_in_u2 = x_out_u1[:, 1:]
        x_in_u2 = torch.cat((x, x_in_u2), 1)

        x10 = self.u_net1.inc(x_in_u2)
        x11 = self.u_net1.down1(x10)
        x12 = self.u_net1.down2(x11)
        x13 = self.u_net1.down3(x12)
        x14 = self.u_net1.down4(x13)
        x15 = self.u_net1.up1(x14, x13)
        x16 = self.u_net1.up2(x15, x12)
        x17 = self.u_net1.up3(x16, x11)
        x18 = self.u_net1.up4(x17, x10)
        x_out_u2 = self.u_net1.outc(x18)

        x_out = torch.cat((x_out_u2, bg), 1)
        """Apply the non-linearities in the last layer."""
        p = x_out[:, [0]]
        i = x_out[:, [1]]
        xyz = x_out[:, 2:5]

        if not self.training:
            p = self.p_nl_inference(p)

        i = self.i_nl(i)
        xyz = self.xyz_nl(xyz)

        if self.n_classes == 5:
            x_out = torch.cat((p, i, xyz), 1)
        elif self.n_classes == 6:
            bg = self.bg_nl(bg)
            x_out = torch.cat((p, i, xyz, bg), 1)
        else:
            raise NotImplementedError("This model is only suitable for 5 or 6 channel output.")
        return x_out


class DoubleOffsetUNetDivided(nn.Module):

    def __init__(self, n_channels, n_classes):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.u_nets = []
        self.u_net0 = UNet(n_channels=n_channels, n_classes=1)
        self.u_net1 = UNet(n_channels=n_channels+1, n_classes=n_classes-1)

        self.p_nl_inference = torch.sigmoid  # identity function since sigmoid is now in loss.
        self.i_nl = torch.sigmoid
        self.xyz_nl = torch.tanh
        self.bg_nl = torch.sigmoid

    def apply_pnl(self, output):
        output[:, [0]] = self.p_nl_inference(output[:, [0]])
        return output

    def forward(self, x):
        """

        :param x:
        :return:
        """
        x1 = self.u_net0.inc(x)
        x2 = self.u_net0.down1(x1)
        x3 = self.u_net0.down2(x2)
        x4 = self.u_net0.down3(x3)
        x5 = self.u_net0.down4(x4)
        x6 = self.u_net0.up1(x5, x4)
        x7 = self.u_net0.up2(x6, x3)
        x8 = self.u_net0.up3(x7, x2)
        x9 = self.u_net0.up4(x8, x1)
        x_out_u1 = self.u_net0.outc(x9)

        bg = x_out_u1[:, [0]]
        x_in_u2 = torch.cat((x, bg.detach().clone()), 1)

        x10 = self.u_net1.inc(x_in_u2)
        x11 = self.u_net1.down1(x10)
        x12 = self.u_net1.down2(x11)
        x13 = self.u_net1.down3(x12)
        x14 = self.u_net1.down4(x13)
        x15 = self.u_net1.up1(x14, x13)
        x16 = self.u_net1.up2(x15, x12)
        x17 = self.u_net1.up3(x16, x11)
        x18 = self.u_net1.up4(x17, x10)
        x_out_u2 = self.u_net1.outc(x18)

        x_out = torch.cat((x_out_u2, bg), 1)
        """Apply the non-linearities in the last layer."""
        p = x_out[:, [0]]
        i = x_out[:, [1]]
        xyz = x_out[:, 2:5]

        if not self.training:
            p = self.p_nl_inference(p)

        i = self.i_nl(i)
        xyz = self.xyz_nl(xyz)

        if self.n_classes == 5:
            x_out = torch.cat((p, i, xyz), 1)
        elif self.n_classes == 6:
            bg = self.bg_nl(bg)
            x_out = torch.cat((p, i, xyz, bg), 1)
        else:
            raise NotImplementedError("This model is only suitable for 5 or 6 channel output.")
        return x_out


class OffsetUnet(UNet):
    def __init__(self, n_channels, n_classes=5):
        super().__init__(n_channels=n_channels, n_classes=n_classes)
        # p non-linearity is in loss (BCEWithLogitsLoss)
        self.p_nl_inference = torch.sigmoid  # identity function since sigmoid is now in loss.
        self.i_nl = torch.sigmoid
        self.xyz_nl = torch.tanh
        self.bg_nl = torch.sigmoid

    def apply_pnl(self, output):
        """
        Apply the non-linearity in the p-channel.
        As this is part of the loss this is usually only done if self.training is
        False or if one wants to do it manually.

        :param output:
        :return:
        """
        output[:, [0]] = self.p_nl_inference(output[:, [0]])
        return output

    def forward(self, x):
        """

        :param x: input
        :param pnl: enforce non-linearity in p even if model.eval() was not executed
        :return:
        """
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)

        """Apply the non-linearities in the last layer."""
        p = x[:, [0]]
        i = x[:, [1]]
        xyz = x[:, 2:5]

        if not self.training:
            p = self.p_nl_inference(p)

        i = self.i_nl(i)
        xyz = self.xyz_nl(xyz)

        if self.n_classes == 5:
            x = torch.cat((p, i, xyz), 1)
        elif self.n_classes == 6:
            bg = self.bg_nl(x[:, [5]])
            x = torch.cat((p, i, xyz, bg), 1)
        else:
            raise NotImplementedError("This model is only suitable for 5 or 6 channel output.")

        return x


class OffSetUNetBGBranch(UNet):
    def __init__(self, n_channels, n_classes=6):
        super().__init__(n_channels=n_channels, n_classes=n_classes)
        if self.n_classes != 6:
            raise ValueError("Specifically implemented for 6 classes.")
        self.b0_c0 = double_conv(64, 64)
        self.b0_c1 = nn.Conv2d(64, 5, (3, 3), padding=1)

        self.b1_c0 = double_conv(64, 64)
        self.b1_c1 = nn.Conv2d(64, 1, (3, 3), padding=1)

        # p non-linearity is in loss (BCEWithLogitsLoss)
        self.p_nl_inference = torch.sigmoid  # identity function since sigmoid is now in loss.
        self.i_nl = torch.sigmoid
        self.xyz_nl = torch.tanh
        self.bg_nl = torch.sigmoid

    def apply_pnl(self, output):
        """
        Apply the non-linearity in the p-channel.
        As this is part of the loss this is usually only done if self.training is
        False or if one wants to do it manually.

        :param output:
        :return:
        """
        output[:, [0]] = self.p_nl_inference(output[:, [0]])
        return output

    def forward(self, x):
        """

        :param x: input
        :param pnl: enforce non-linearity in p even if model.eval() was not executed
        :return:
        """
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        """Branching"""
        o0 = self.b0_c0(x)
        o0 = self.b0_c1(o0)

        o1 = self.b1_c0(x)
        o1 = self.b1_c1(o1)

        """Apply the non-linearities in the last layer."""
        p = o0[:, [0]]
        i = o0[:, [1]]
        xyz = o0[:, 2:5]

        if not self.training:
            p = self.p_nl_inference(p)

        i = self.i_nl(i)
        xyz = self.xyz_nl(xyz)

        if self.n_classes == 5:
            x = torch.cat((p, i, xyz), 1)
        elif self.n_classes == 6:
            bg = self.bg_nl(o1[:, [0]])
            x = torch.cat((p, i, xyz, bg), 1)
        else:
            raise NotImplementedError("This model is only suitable for 5 or 6 channel output.")

        return x

if __name__ == '__main__':
    img = torch.rand((2, 3, 32, 32)).cuda()
    test = torch.rand((2, 6, 32, 32)).cuda()

    criterion = torch.nn.MSELoss()
    model = OffSetUNetBGBranch(3, 6).cuda()
    out = model(img)
    loss = criterion(out, test)
    loss.backward()

    print("Done.")
