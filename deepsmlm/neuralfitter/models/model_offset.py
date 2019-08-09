import torch

from deepsmlm.neuralfitter.models.unet_model import *


class OffsetUnet(UNet):
    def __init__(self, n_channels):
        super().__init__(n_channels=n_channels, n_classes=5)
        # p non-linearity is in loss (BCEWithLogitsLoss)
        self.p_nl_inference = torch.sigmoid  # identity function since sigmoid is now in loss.
        self.i_nl = torch.sigmoid
        self.xyz_nl = torch.tanh

    def apply_pnl(self, output):
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
        xyz = x[:, 2:]

        if not self.training:
            p = self.p_nl_inference(p)

        i = self.i_nl(i)
        xyz = self.xyz_nl(xyz)
        x = torch.cat((p, i, xyz), 1)

        return x


if __name__ == '__main__':
    img = torch.rand((1, 1, 32, 32))
    test = torch.rand((1, 1, 32, 32*4, 32*4))

    criterion = torch.nn.MSELoss()
    model = UNet(1, 1, F.relu)
    out = model(img)

    print("Done.")