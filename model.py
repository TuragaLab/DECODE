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

        self.conv5 = nn.Conv2d(512, 128, (3, 3), 1, padding=1)
        self.conv5_bn = nn.BatchNorm2d(128)
        self.conv6 = nn.Conv2d(128, 64, (3, 3), 1, padding=1)
        self.conv6_bn = nn.BatchNorm2d(64)
        self.conv7 = nn.Conv2d(64, 32, (3, 3), 1, padding=1)
        self.conv7_bn = nn.BatchNorm2d(32)
        self.conv8 = nn.Conv2d(32, 1, (1, 1), 1, bias=False)

    def forward(self, x):
        # encode and downsample
        x = F.max_pool2d(self.act(self.conv1_bn(self.conv1(x))), 2, 2)
        x = F.max_pool2d(self.act(self.conv2_bn(self.conv2(x))), 2, 2)
        x = F.max_pool2d(self.act(self.conv3_bn(self.conv3(x))), 2, 2)

        x = F.interpolate(self.act(self.conv4_bn(self.conv4(x))), scale_factor=(2, 2))
        x = F.interpolate(self.act(self.conv5_bn(self.conv5(x))), scale_factor=(2, 2))
        x = F.interpolate(self.act(self.conv6_bn(self.conv6(x))), scale_factor=(2, 2))
        x = self.act(self.conv7_bn(self.conv7(x)))
        x = self.act(self.conv8(x))

        return x


    def weight_init(self):
        for m in self._modules:
            if isinstance(m, nn.Conv2d):
                nn.init.uniform_(m)
