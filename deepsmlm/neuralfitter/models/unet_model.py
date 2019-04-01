# full assembly of the sub-parts to form the complete net

import torch.nn.functional as F

from deepsmlm.neuralfitter.models.unet_parts import *


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, out_func=F.relu):
        super(UNet, self).__init__()
        self.inc = inconv(n_channels, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 512)
        self.up1 = up(1024, 256)
        self.up2 = up(512, 128)
        self.up3 = up(256, 64)
        self.up4 = up(128, 64)
        self.outc = outconv(64, n_classes)
        self.out_func = out_func  # F.sigmoid originally

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, None)
        x = self.up2(x, None)
        x = self.up3(x, None)
        x = self.up4(x, None)
        x = self.outc(x)
        return self.out_func(x)


class UNetLearned3D(UNet):
    def __init__(self, n_channels):
        super().__init__(n_channels, 1, F.relu)
        self.inc = inconv_3d(n_channels, 64)
        self.down1 = down_3d(64, 128)
        self.down2 = down_3d(128, 256)
        self.down3 = down_3d(256, 512)
        self.down4 = down_3d(512, 512)
        self.up1 = up_3d(512, 256, scale_factor=(2, 2, 2))
        self.up2 = up_3d(256, 128, scale_factor=(4, 2, 2))
        self.up3 = up_3d(128, 64, scale_factor=(2, 2, 2))
        self.up4 = up_3d(64, 32, scale_factor=(2, 2, 2))
        self.outc = outconv_3d(32, n_channels)

    def forward(self, x):
        """N x C x H x W --> N x C x D x H x W
        Unsqueeze to add singelton depth dimension."""
        x = x.unsqueeze(2)
        return super().forward(x)


if __name__ == '__main__':
    img = torch.rand((1, 1, 32, 32))
    test = torch.rand((1, 1, 32, 32*4, 32*4)).cuda()

    criterion = torch.nn.MSELoss()
    model = UNet(1, 1, F.relu)
    # out = model(img)

    model = UNetLearned3D(1)
    model = model.cuda()
    img = img.cuda()
    img = F.interpolate(img, scale_factor=(4, 4))
    out = model(img)
    loss = criterion(out, test)
    loss.backward()

    print('Success.')
