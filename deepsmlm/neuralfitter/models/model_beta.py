import torch
import torch.nn as nn
import torch.nn.functional as F

from deepsmlm.neuralfitter.models.model_densenet import DenseNet
from deepsmlm.neuralfitter.models.model import DeepSMLN, ResNet


class EncoderFC(DeepSMLN):
    def __init__(self, limits):
        super().__init__(1)
        self.limits = limits
        self.fc_net = ResNet(512, 512, 3)
        self.fc_out = nn.Linear(512, 1)

    def forward(self, x):
        x = super().encode(x)
        x = x.view(x.shape[0], -1)
        x = self.fc_net.forward(x)
        # x = torch.sigmoid(self.fc_out(x)) * (self.limits[1] - self.limits[0]) + self.limits[0]
        x = self.fc_out(x)
        return x

    def weight_init(self):
        # for m in self.modules():
        #     m.weight_init()
        super().weight_init()
        self.fc_net.weight_init()
        nn.init.xavier_uniform_(self.fc_out.weight, gain=1000)


class DenseNetResNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.densnet_features = 64
        self.conv_net = DenseNet(num_classes=self.densnet_features, num_channels=1)
        self.fc_net = SuperDumbFCNet(self.densnet_features, None)
        # self.fc_net = ResNet(self.densnet_features, self.densnet_features, 2)
        self.fc_out = nn.Linear(self.densnet_features, 1)

    def forward(self, x):
        x = self.conv_net.forward(x)
        x = self.fc_net.forward(x)
        x = self.fc_out.forward(x)
        return x


class SuperDumbFCNet(nn.Module):
    def __init__(self, dim_hidden, limits):
        super().__init__()
        self.limits = limits
        self.act = F.relu

        self.fc1 = nn.Linear(dim_hidden, dim_hidden)
        self.bn1 = nn.BatchNorm1d(dim_hidden)
        self.fc2 = nn.Linear(dim_hidden, dim_hidden)
        self.bn2 = nn.BatchNorm1d(dim_hidden)
        self.fc3 = nn.Linear(dim_hidden, dim_hidden)
        self.bn3 = nn.BatchNorm1d(dim_hidden)
        self.fc4 = nn.Linear(dim_hidden, dim_hidden)
        self.bn4 = nn.BatchNorm1d(dim_hidden)
        self.fc5 = nn.Linear(dim_hidden, dim_hidden)
        self.bn5 = nn.BatchNorm1d(dim_hidden)
        self.fc6 = nn.Linear(dim_hidden, dim_hidden)
        self.bn6 = nn.BatchNorm1d(dim_hidden)
        self.fc_out = nn.Linear(dim_hidden, 1)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.bn1(self.act(self.fc1(x)))
        x = self.bn2(self.act(self.fc2(x)))
        x = self.bn3(self.act(self.fc3(x)))
        x = self.bn4(self.act(self.fc4(x)))
        x = self.bn5(self.act(self.fc5(x)))
        x = self.bn6(self.act(self.fc6(x)))
        z = self.fc_out(x) * self.limits[1]
        # x = torch.sigmoid(self.fc_out(x)) * (self.limits[1] - self.limits[0]) + self.limits[0]
        xy = (torch.ones((z.shape[0], 2)) * torch.tensor([8., 8.])).to(z.device)
        xyz = torch.cat((xy, z), dim=1).unsqueeze(1)
        return xyz, torch.ones_like(xyz[:, :, 0])

    def weight_init(self):
        print("Not implemented.")


"""From https://github.com/kuangliu/pytorch-cifar/blob/master/models/lenet.py"""
class LeNetModified(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 6, 5)
        self.conv1_bn = nn.BatchNorm2d(6)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.conv2_bn = nn.BatchNorm2d(16)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.bn1 = nn.BatchNorm1d(120)
        self.fc2 = nn.Linear(120, 84)
        self.bn2 = nn.BatchNorm1d(84)
        self.fc3 = nn.Linear(84, 32)
        self.bn3 = nn.BatchNorm1d(32)
        self.fc4 = nn.Linear(32, 1)

    def forward(self, x):
        out = self.conv1_bn(F.relu(self.conv1(x)))
        out = F.max_pool2d(out, 2)
        out = self.conv2_bn(F.relu(self.conv2(out)))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = self.bn1(F.relu(self.fc1(out)))
        out = self.bn2(F.relu(self.fc2(out)))
        out = self.bn3(F.relu(self.fc3(out)))
        out = self.fc4(out)
        return out

    def weight_init(self):
        print("No Weight init implemented.")



"""From https://github.com/jaxony/unet-pytorch/blob/master/model.py"""


def conv3x3(in_channels, out_channels, stride=1,
            padding=1, bias=True, groups=1):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        padding=padding,
        bias=bias,
        groups=groups)


def upconv2x2(in_channels, out_channels, mode='transpose'):
    if mode == 'transpose':
        return nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=2,
            stride=2)
    else:
        # out_channels is always going to be the same
        # as in_channels
        return nn.Sequential(
            nn.Upsample(mode='bilinear', scale_factor=2),
            conv1x1(in_channels, out_channels))


def conv1x1(in_channels, out_channels, groups=1):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=1,
        groups=groups,
        stride=1)


class DownConv(nn.Module):
    """
    A helper Module that performs 2 convolutions and 1 MaxPool.
    A ReLU activation follows each convolution.
    """

    def __init__(self, in_channels, out_channels, pooling=True):
        super(DownConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.pooling = pooling

        self.conv1 = conv3x3(self.in_channels, self.out_channels)
        self.conv2 = conv3x3(self.out_channels, self.out_channels)

        if self.pooling:
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        before_pool = x
        if self.pooling:
            x = self.pool(x)
        return x, before_pool


class UpConv(nn.Module):
    """
    A helper Module that performs 2 convolutions and 1 UpConvolution.
    A ReLU activation follows each convolution.
    """

    def __init__(self, in_channels, out_channels,
                 merge_mode='concat', up_mode='transpose'):
        super(UpConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.merge_mode = merge_mode
        self.up_mode = up_mode

        self.upconv = upconv2x2(self.in_channels, self.out_channels,
                                mode=self.up_mode)

        if self.merge_mode == 'concat':
            self.conv1 = conv3x3(
                2 * self.out_channels, self.out_channels)
        else:
            # num of input channels to conv2 is same
            self.conv1 = conv3x3(self.out_channels, self.out_channels)
        self.conv2 = conv3x3(self.out_channels, self.out_channels)

    def forward(self, from_down, from_up):
        """ Forward pass
        Arguments:
            from_down: tensor from the encoder pathway
            from_up: upconv'd tensor from the decoder pathway
        """
        from_up = self.upconv(from_up)
        if self.merge_mode == 'concat':
            x = torch.cat((from_up, from_down), 1)
        else:
            x = from_up + from_down
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        return x


class UNet(nn.Module):
    """ `UNet` class is based on https://arxiv.org/abs/1505.04597

    The U-Net is a convolutional encoder-decoder neural network.
    Contextual spatial information (from the decoding,
    expansive pathway) about an input tensor is merged with
    information representing the localization of details
    (from the encoding, compressive pathway).

    Modifications to the original paper:
    (1) padding is used in 3x3 convolutions to prevent loss
        of border pixels
    (2) merging outputs does not require cropping due to (1)
    (3) residual connections can be used by specifying
        UNet(merge_mode='add')
    (4) if non-parametric upsampling is used in the decoder
        pathway (specified by upmode='upsample'), then an
        additional 1x1 2d convolution occurs after upsampling
        to reduce channel dimensionality by a factor of 2.
        This channel halving happens with the convolution in
        the tranpose convolution (specified by upmode='transpose')
    """

    def __init__(self, num_classes, in_channels=3, depth=5,
                 start_filts=64, up_mode='transpose',
                 merge_mode='concat'):
        """
        Arguments:
            in_channels: int, number of channels in the input tensor.
                Default is 3 for RGB images.
            depth: int, number of MaxPools in the U-Net.
            start_filts: int, number of convolutional filters for the
                first conv.
            up_mode: string, type of upconvolution. Choices: 'transpose'
                for transpose convolution or 'upsample' for nearest neighbour
                upsampling.
        """
        super(UNet, self).__init__()

        if up_mode in ('transpose', 'upsample'):
            self.up_mode = up_mode
        else:
            raise ValueError("\"{}\" is not a valid mode for "
                             "upsampling. Only \"transpose\" and "
                             "\"upsample\" are allowed.".format(up_mode))

        if merge_mode in ('concat', 'add'):
            self.merge_mode = merge_mode
        else:
            raise ValueError("\"{}\" is not a valid mode for"
                             "merging up and down paths. "
                             "Only \"concat\" and "
                             "\"add\" are allowed.".format(up_mode))

        # NOTE: up_mode 'upsample' is incompatible with merge_mode 'add'
        if self.up_mode == 'upsample' and self.merge_mode == 'add':
            raise ValueError("up_mode \"upsample\" is incompatible "
                             "with merge_mode \"add\" at the moment "
                             "because it doesn't make sense to use "
                             "nearest neighbour to reduce "
                             "depth channels (by half).")

        self.num_classes = num_classes
        self.in_channels = in_channels
        self.start_filts = start_filts
        self.depth = depth

        self.down_convs = []
        self.up_convs = []

        # create the encoder pathway and add to a list
        for i in range(depth):
            ins = self.in_channels if i == 0 else outs
            outs = self.start_filts * (2 ** i)
            pooling = True if i < depth - 1 else False

            down_conv = DownConv(ins, outs, pooling=pooling)
            self.down_convs.append(down_conv)

        # create the decoder pathway and add to a list
        # - careful! decoding only requires depth-1 blocks
        for i in range(depth - 1):
            ins = outs
            outs = ins // 2
            up_conv = UpConv(ins, outs, up_mode=up_mode,
                             merge_mode=merge_mode)
            self.up_convs.append(up_conv)

        self.conv_final = conv1x1(outs, self.num_classes)

        # add the list of modules to current module
        self.down_convs = nn.ModuleList(self.down_convs)
        self.up_convs = nn.ModuleList(self.up_convs)

        self.reset_params()

    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_normal_(m.weight)
            nn.init.constant_(m.bias, 0)

    def reset_params(self):
        for i, m in enumerate(self.modules()):
            self.weight_init(m)

    def forward(self, x):
        encoder_outs = []

        # encoder pathway, save outputs for merging
        for i, module in enumerate(self.down_convs):
            x, before_pool = module(x)
            encoder_outs.append(before_pool)

        for i, module in enumerate(self.up_convs):
            before_pool = encoder_outs[-(i + 2)]
            x = module(before_pool, x)

        # No softmax is used. This means you need to use
        # nn.CrossEntropyLoss is your training script,
        # as this module includes a softmax already.
        x = self.conv_final(x)
        return x


if __name__ == "__main__":
    """
    Testing.
    """
    model = UNet(num_classes=1, in_channels=1, depth=5, merge_mode='concat')
    x = torch.rand((1, 1, 320, 320), requires_grad=True)
    out = model(x)
    loss = torch.sum(out)
    loss.backward()

    model = SuperDumbFCNet(256, (-1., 1.))
    x = torch.rand((32, 1, 16, 16), requires_grad=True)
    out = model(x)
    loss = torch.sum(out)
    loss.backward()

    model = EncoderFC((-500., 500.))
    x = torch.rand((32, 1, 26, 26), requires_grad=True)
    out = model(x)
    loss = torch.sum(out)
    loss.backward()

    model = DenseNetResNet()
    x = torch.rand((32, 1, 26, 26), requires_grad=True)
    out = model(x)
    loss = torch.sum(out)
    loss.backward()

    model = DenseNet(num_channels=1)
    x = torch.rand((32, 1, 26, 26), requires_grad=True)
    out = model(x)

    print("Sucess.")
