import torch
import torch.nn as nn
import torch.nn.functional as F


class DeepSLMN(nn.Module):
    """
    Architecture Nehme, E. - Deep-STORM - https://www.osapublishing.org/abstract.cfm?URI=optica-5-4-458
    """
    def __init__(self):
        super().__init__()

        self.act = F.relu

        # encode
        self.conv1 = nn.Conv2d(1, 32, (3, 3), 1, padding=1)
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
        self.conv7 = nn.Conv2d(64 + 1, 32, (3, 3), 1, padding=1)
        self.conv7_bn = nn.BatchNorm2d(32)
        self.conv8 = nn.Conv2d(32, 1, (1, 1), 1, bias=False)

    def forward(self, x0):
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
                nn.init.uniform_(m)
