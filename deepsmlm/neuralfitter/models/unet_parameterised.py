# https://raw.githubusercontent.com/jvanvugt/pytorch-unet/master/unet.py
# Adapted from https://discuss.pytorch.org/t/unet-implementation/426

import torch
from torch import nn
import torch.nn.functional as F


class MUNet(nn.Module):
    def __init__(self, in_channels, n_classes, depth, wf, mlt_head, branch_depth, branch_filters, padding, batch_norm,
                 up_mode, activation, last_activation):
        super().__init__()
        assert mlt_head in ('plain', 'branched')
        assert last_activation in (True, False)

        self.in_channels = in_channels
        self.n_classes = n_classes
        self.mlt_head = mlt_head
        self.last_layer = None
        self.net = nn.ModuleList()
        if mlt_head == 'plain':
            unet_classes = n_classes
        elif mlt_head == 'branched':
            unet_classes = 2**wf

        self.unet = UNet(in_channels=in_channels,
                         n_classes=unet_classes,
                         depth=depth,
                         wf=wf,
                         padding=padding,
                         batch_norm=batch_norm,
                         up_mode=up_mode,
                         activation=activation)

        if mlt_head == 'branched':
            self.head = nn.ModuleList()
            for _ in range(n_classes):
                self.head.append(self._build_task_block(unet_classes, branch_depth, branch_filters, activation))

        self.p_nl = torch.sigmoid
        self.phot_nl = torch.sigmoid
        self.xyz_nl = torch.tanh
        self.bg_nl = torch.sigmoid

    @staticmethod
    def parse(param):
        return MUNet(in_channels=param['HyperParameter']['channels_in'],
                     n_classes=param['HyperParameter']['channels_out'],
                     depth=param['HyperParameter']['arch_param']['depth'],
                     wf=param['HyperParameter']['arch_param']['wf'],
                     mlt_head=param['HyperParameter']['arch_param']['mlt_head'],
                     branch_depth=param['HyperParameter']['arch_param']['branch_depth'],
                     branch_filters=param['HyperParameter']['arch_param']['branch_filters'],
                     padding=param['HyperParameter']['arch_param']['padding'],
                     batch_norm=param['HyperParameter']['arch_param']['batch_norm'],
                     up_mode=param['HyperParameter']['arch_param']['up_mode'],
                     activation=eval(param['HyperParameter']['arch_param']['activation']),
                     last_activation=param['HyperParameter']['arch_param']['last_activation'])

    @staticmethod
    def _build_task_block(in_ch, depth, filters, activation):
        head = []
        head.append(nn.Conv2d(in_ch, filters, kernel_size=3, padding=1))
        head.append(activation)
        for i in range(depth - 1):
            head.append(nn.Conv2d(filters, filters, kernel_size=3, padding=1))
            head.append(activation)

        head.append(nn.Conv2d(filters, 1, kernel_size=1))
        head = nn.Sequential(*head)
        return head

    def apply_pnl(self, out):
        out[:, [0]] = self.p_nl(out[:, [0]])
        return out

    def forward(self, x):
        o = self.unet.forward(x)

        if self.mlt_head == 'branched':
            o_head = []
            for branch in self.head:
                o_head.append(branch.forward(o))
            o = torch.cat(o_head, 1)

        p = o[:, [0]]
        i = o[:, [1]]
        xyz = o[:, 2:5]
        if not self.training:
            p = self.p_nl(p)

        if self.last_layer:
            i = self.phot_nl(i)
            xyz = self.xyz_nl(xyz)

        if self.n_classes == 5:
            o = torch.cat((p, i, xyz), 1)
        elif self.n_classes == 6:
            bg = self.bg_nl(o[:, [5]])
            o = torch.cat((p, i, xyz, bg), 1)
        else:
            raise NotImplementedError("This model is only suitable for 5 or 6 channel output.")

        return o
        

class UNet(nn.Module):
    def __init__(
        self,
        in_channels=1,
        n_classes=2,
        depth=5,
        wf=6,
        padding=False,
        batch_norm=False,
        up_mode='upconv',
        activation=nn.ReLU()
    ):
        """
        Implementation of
        U-Net: Convolutional Networks for Biomedical Image Segmentation
        (Ronneberger et al., 2015)
        https://arxiv.org/abs/1505.04597

        Using the default arguments will yield the exact version used
        in the original paper

        Args:
            in_channels (int): number of input channels
            n_classes (int): number of output channels
            depth (int): depth of the network
            wf (int): number of filters in the first layer is 2**wf
            padding (bool): if True, apply padding such that the input shape
                            is the same as the output.
                            This may introduce artifacts
            batch_norm (bool): Use BatchNorm after layers with an
                               activation function
            up_mode (str): one of 'upconv' or 'upsample'.
                           'upconv' will use transposed convolutions for
                           learned upsampling.
                           'upsample' will use bilinear upsampling.
        """
        super(UNet, self).__init__()
        assert up_mode in ('upconv', 'upsample')
        self.padding = padding
        self.depth = depth
        prev_channels = in_channels
        self.activation = activation
        self.down_path = nn.ModuleList()
        for i in range(depth):
            self.down_path.append(
                UNetConvBlock(prev_channels, 2 ** (wf + i), padding, batch_norm, activation)
            )
            prev_channels = 2 ** (wf + i)

        self.up_path = nn.ModuleList()
        for i in reversed(range(depth - 1)):
            self.up_path.append(
                UNetUpBlock(prev_channels, 2 ** (wf + i), up_mode, padding, batch_norm, activation)
            )
            prev_channels = 2 ** (wf + i)

        self.last = nn.Conv2d(prev_channels, n_classes, kernel_size=1)

    def forward(self, x):
        blocks = []
        for i, down in enumerate(self.down_path):
            x = down(x)
            if i != len(self.down_path) - 1:
                blocks.append(x)
                x = F.max_pool2d(x, 2)

        for i, up in enumerate(self.up_path):
            x = up(x, blocks[-i - 1])

        return self.last(x)


class UNetConvBlock(nn.Module):
    def __init__(self, in_size, out_size, padding, batch_norm, activation):
        super(UNetConvBlock, self).__init__()
        block = []

        block.append(nn.Conv2d(in_size, out_size, kernel_size=3, padding=int(padding)))
        block.append(activation)
        if batch_norm:
            block.append(nn.BatchNorm2d(out_size))

        block.append(nn.Conv2d(out_size, out_size, kernel_size=3, padding=int(padding)))
        block.append(activation)
        if batch_norm:
            block.append(nn.BatchNorm2d(out_size))

        self.block = nn.Sequential(*block)

    def forward(self, x):
        out = self.block(x)
        return out


class UNetUpBlock(nn.Module):
    def __init__(self, in_size, out_size, up_mode, padding, batch_norm, activation):
        super(UNetUpBlock, self).__init__()
        if up_mode == 'upconv':
            self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2)
        elif up_mode == 'upsample':
            self.up = nn.Sequential(
                nn.Upsample(mode='bilinear', scale_factor=2, align_corners=False),
                nn.Conv2d(in_size, out_size, kernel_size=1),
            )

        self.conv_block = UNetConvBlock(in_size, out_size, padding, batch_norm, activation)

    def center_crop(self, layer, target_size):
        _, _, layer_height, layer_width = layer.size()
        diff_y = (layer_height - target_size[0]) // 2
        diff_x = (layer_width - target_size[1]) // 2
        return layer[
            :, :, diff_y : (diff_y + target_size[0]), diff_x : (diff_x + target_size[1])
        ]

    def forward(self, x, bridge):
        up = self.up(x)
        # crop1 = self.center_crop(bridge, up.shape[2:])
        crop1 = bridge
        out = torch.cat([up, crop1], 1)
        out = self.conv_block(out)

        return out