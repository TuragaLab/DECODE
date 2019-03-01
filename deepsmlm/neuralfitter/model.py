import torch
import torch.nn as nn
import torch.nn.functional as F


class DeepSLMN(nn.Module):
    """
    Architecture Nehme, E. - Deep-STORM - https://www.osapublishing.org/abstract.cfm?URI=optica-5-4-458
    """
    def __init__(self):
        super().__init__()

        self.upsampling = 8  # as parameter? or constant here?
        self.act = F.relu

        in_ch = 3
        # encode
        self.conv1 = nn.Conv2d(in_ch, 32, (3, 3), 1, padding=1)
        self.conv1_bn = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, (3, 3), 1, padding=1)
        self.conv2_bn = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, (3, 3), 1, padding=1)
        self.conv3_bn = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 512, (3, 3), 1, padding=1)
        self.conv4_bn = nn.BatchNorm2d(512)

        self.conv5 = nn.Conv2d(512 + 64, 128, (3, 3), 1, padding=1)
        self.conv5_bn = nn.BatchNorm2d(128)
        self.conv6 = nn.Conv2d(128 + 32, 64, (3, 3), 1, padding=1)
        self.conv6_bn = nn.BatchNorm2d(64)
        self.conv7 = nn.Conv2d(64 + in_ch, 32, (3, 3), 1, padding=1)
        self.conv7_bn = nn.BatchNorm2d(32)
        self.conv8 = nn.Conv2d(32, 1, (1, 1), 1, bias=False)

    def forward(self, x0):
        """
        F.interpolate uses the last 2/3/4/5/ dimensions for interpolation,
        so unsqueeze and squeeze a bit.
        """
        x0 = F.interpolate(x0, scale_factor=self.upsampling, mode='nearest')

        # encode and downsample
        x1 = F.max_pool2d(self.act(self.conv1_bn(self.conv1(x0))), 2, 2)
        x2 = F.max_pool2d(self.act(self.conv2_bn(self.conv2(x1))), 2, 2)
        x3 = F.max_pool2d(self.act(self.conv3_bn(self.conv3(x2))), 2, 2)

        x4 = F.interpolate(self.act(self.conv4_bn(self.conv4(x3))), scale_factor=(2, 2))

        x4 = torch.cat((x4, x2), dim=1)
        x5 = F.interpolate(self.act(self.conv5_bn(self.conv5(x4))), scale_factor=(2, 2))

        x5 = torch.cat((x5, x1), dim=1)
        x6 = F.interpolate(self.act(self.conv6_bn(self.conv6(x5))), scale_factor=(2, 2))

        x6 = torch.cat((x6, x0), dim=1)
        x7 = self.act(self.conv7_bn(self.conv7(x6)))
        x8 = self.act(self.conv8(x7))

        return x8

    def weight_init(self):
        for m in self._modules:
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m, gain=1)


# Based on deep_loco by Boyd
class DeepLoco(nn.Module):
    def __init__(self, extent, ch_in=1, dim_out=2):
        super().__init__()
        self.feature_net = DeepConvNet(ch_in)
        self.fc_net = ResNet(256 * 4 * 4, 2048, 2)
        self.phot_xyz_net = PhotXYZnet(2048, 256, extent[0], extent[1], extent[2], dim_out)

    def forward(self, x):
        return self.phot_xyz_net(self.fc_net(self.feature_net(x)))

    def weight_init(self):
        print('Not implemented.')


class DeepConvNet(nn.Module):
    def __init__(self, dim_in=1):
        super().__init__()
        self.conv1 = nn.Conv2d(dim_in, 16, 5, padding=(5-1)//2)
        self.conv2 = nn.Conv2d(16, 16, 5, padding=(5-1)//2)

        self.conv_d1 = nn.Conv2d(16, 64, 2, stride=2)
        self.conv3 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv4 = nn.Conv2d(64, 64, 3, padding=1)

        self.conv_d2 = nn.Conv2d(64, 256, 2, stride=2)
        self.conv5 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv6 = nn.Conv2d(256, 256, 3, padding=1)

        self.conv_d3 = nn.Conv2d(256, 256, 4, stride=4)

    def forward(self, x):
        x = self.conv2(self.conv1(x))
        x = self.conv4(self.conv3(self.conv_d1(x)))
        x = self.conv6(self.conv5(self.conv_d2(x)))
        x = self.conv_d3(x)
        return x


class ResNet(nn.Module):
    def __init__(self, dim_in, dim_hidden, depth):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_hidden
        self.depth = depth

        self.linear_init = nn.Linear(dim_in, dim_hidden)
        self.residual_blocks = nn.ModuleList([ResidualBlock(dim_hidden)] * depth)

    def forward(self, x):
        x = x.view(x.shape[0], -1)

        x = self.linear_init(x)
        for rb in self.residual_blocks:
            x = rb(x)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.l1 = nn.Linear(dim, dim)
        self.l2 = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        return x + self.l2(F.relu(self.l1(x)))


class PhotXYZnet(nn.Module):
    def __init__(self, dim_in, max_num_emitter, xextent, yextent, zextent, emitter_dim=3):
        super().__init__()

        self.xextent = xextent
        self.yextent = yextent
        self.zextent = zextent

        self.scale_tensor = torch.tensor([self.xextent[1] - self.xextent[0],
                                          self.yextent[1] - self.yextent[0],
                                          self.zextent[1] - self.zextent[0]]).cuda()
        self.shift_tensor = torch.tensor([self.xextent[0], self.yextent[0], self.zextent[0]]).cuda()
        self.emitter_dim = emitter_dim
        self.photon_fcnet = nn.Linear(dim_in, max_num_emitter)
        self.xyz_fcnet = nn.Linear(dim_in, max_num_emitter * emitter_dim)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        """Place xyz in apropriate limits. I am unhappy about this."""
        xyz = torch.sigmoid(self.xyz_fcnet(x))
        xyz = xyz.view(x.shape[0], -1, self.emitter_dim)
        xyz = xyz * self.scale_tensor + self.shift_tensor

        phot = F.relu(self.photon_fcnet(x))
        return xyz, phot
