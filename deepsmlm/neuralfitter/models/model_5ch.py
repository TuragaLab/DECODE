import torch
import torch.nn as nn
import torch.nn.functional as functional


from deepsmlm.neuralfitter.models.resnet import ResNet, BasicBlock


class ResNetModified(ResNet):
    def __init__(self, block, layers,
                 zero_init_residual=False,
                 groups=1,
                 width_per_group=64,
                 replace_stride_with_dilation=None,
                 norm_layer=None):

        super().__init__(block, layers,
                         num_classes=100,
                         zero_init_residual=zero_init_residual,
                         groups=groups,
                         width_per_group=width_per_group,
                         replace_stride_with_dilation=replace_stride_with_dilation,
                         norm_layer=norm_layer)

        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=1,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1,
                                       dilate=replace_stride_with_dilation[2])

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x


if __name__ == '__main__':

    rnet = ResNetModified(BasicBlock, [2, 2, 2, 2])

    x = torch.rand((2, 3, 64, 64), requires_grad=True)
    out = rnet.forward(x)
    l = out.sum().backward()

    print("Done.")
