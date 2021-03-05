import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings

# by constantin pape
# from https://github.com/constantinpape/mu-net


def get_activation(activation):
    """ Get activation from str or nn.Module
    """
    if activation is None:
        return None
    elif isinstance(activation, str):
        activation = getattr(nn, activation)()
    else:
        activation = activation()
        assert isinstance(activation, nn.Module)
    return activation


class Upsample(nn.Module):
    """ Upsample the input and change the number of channels
    via 1x1 Convolution if a different number of input/output channels is specified.
    """

    def __init__(self, scale_factor, mode='nearest',
                 in_channels=None, out_channels=None, align_corners=False,
                 ndim=3):
        super().__init__()
        self.mode = mode
        self.scale_factor = scale_factor
        self.align_corners = align_corners
        if in_channels != out_channels:
            if ndim == 2:
                self.conv = nn.Conv2d(in_channels, out_channels, 1)
            elif ndim == 3:
                self.conv = nn.Conv3d(in_channels, out_channels, 1)
            else:
                raise ValueError("Only 2d and 3d supported")
        else:
            self.conv = None

    def forward(self, input):
        x = F.interpolate(input, scale_factor=self.scale_factor, mode=self.mode, align_corners=self.align_corners)
        if self.conv is not None:
            return self.conv(x)
        else:
            return x


# TODO implement side outputs
class UNetBase(nn.Module):
    """ UNet Base class implementation

    Deriving classes must implement
    - _conv_block(in_channels, out_channels, level, part)
        return conv block for a U-Net level
    - _pooler(level)
        return pooling operation used for downsampling in-between encoders
    - _upsampler(in_channels, out_channels, level)
        return upsampling operation used for upsampling in-between decoders
    - _out_conv(in_channels, out_channels)
        return output conv layer

    Arguments:
      in_channels: number of input channels
      out_channels: number of output channels
      depth: depth of the network
      initial_features: number of features after first convolution
      gain: growth factor of features
      pad_convs: whether to use padded convolutions
      norm: whether to use batch-norm, group-norm or None
      p_dropout: dropout probability
      final_activation: activation applied to the network output
    """
    norms = ('BatchNorm', 'GroupNorm')
    pool_modules = ('MaxPool', 'StrideConv')

    def __init__(self, in_channels, out_channels, depth=4, initial_features=64, gain=2, pad_convs=False, norm=None,
                 norm_groups=None, p_dropout=None, final_activation=None, activation=nn.ReLU(), pool_mode='MaxPool',
                 skip_gn_level=None, upsample_mode='bilinear'):
        super().__init__()

        self.depth = depth
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.pad_convs = pad_convs
        if norm is not None:
            assert norm in self.norms
        assert pool_mode in self.pool_modules
        self.pool_mode = pool_mode
        self.norm = norm
        self.norm_groups = norm_groups
        if p_dropout is not None:
            assert isinstance(p_dropout, (float, dict))
        self.p_dropout = p_dropout
        self.skip_gn_level=skip_gn_level

        # modules of the encoder path
        n_features = [in_channels] + [initial_features * gain ** level
                                      for level in range(self.depth)]
        self.features_per_level = n_features
        self.encoder = nn.ModuleList([self._conv_block(n_features[level], n_features[level + 1],
                                                       level, part='encoder', activation=activation)
                                      for level in range(self.depth)])

        # the base convolution block
        self.base = self._conv_block(n_features[-1], gain * n_features[-1],
                                     part='base', level=depth, activation=activation)

        # modules of the decoder path
        n_features = [initial_features * gain ** level
                      for level in range(self.depth + 1)]
        n_features = n_features[::-1]
        self.decoder = nn.ModuleList([self._conv_block(n_features[level], n_features[level + 1],
                                                       self.depth - level - 1, part='decoder', activation=activation)
                                      for level in range(self.depth)])

        # the pooling layers;
        self.poolers = nn.ModuleList([self._pooler(level + 1) for level in range(self.depth)])
        # the upsampling layers
        self.upsamplers = nn.ModuleList([self._upsampler(n_features[level],
                                                         n_features[level + 1],
                                                         self.depth - level - 1,
                                                         mode=upsample_mode)
                                         for level in range(self.depth)])
        # output conv and activation
        # the output conv is not followed by a non-linearity, because we apply
        # activation afterwards
        self.out_conv = self._out_conv(n_features[-1], out_channels)
        self.activation = get_activation(final_activation)

    @staticmethod
    def _crop_tensor(input_, shape_to_crop):
        input_shape = input_.shape
        # get the difference between the shapes
        shape_diff = tuple((ish - csh) // 2
                           for ish, csh in zip(input_shape, shape_to_crop))
        # if input_.size() == shape_to_crop:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if all(sd == 0 for sd in shape_diff):
                return input_
        # calculate the crop
        crop = tuple(slice(sd, sh - sd)
                     for sd, sh in zip(shape_diff, input_shape))
        return input_[crop]

    # crop the `from_encoder` tensor and concatenate both
    def _crop_and_concat(self, from_decoder, from_encoder):
        cropped = self._crop_tensor(from_encoder, from_decoder.shape)
        return torch.cat((cropped, from_decoder), dim=1)

    def forward_parts(self, parts):
        if 'encoder' in parts:
            x = input
            # apply encoder path
            encoder_out = []
            for level in range(self.depth):
                x = self.encoder[level](x)
                encoder_out.append(x)
                x = self.poolers[level](x)

        if 'base' in parts:
            x = self.base(x)

        if 'decoder' in parts:
            # apply decoder path
            encoder_out = encoder_out[::-1]
            for level in range(self.depth):
                x = self.upsamplers[level](x)
                x = self.decoder[level](self._crop_and_concat(x,
                                                              encoder_out[level]))

            # apply output conv and activation (if given)
            x = self.out_conv(x)
            if self.activation is not None:
                x = self.activation(x)

        return x

    def forward(self, input):
        x = input
        # apply encoder path
        encoder_out = []
        for level in range(self.depth):
            x = self.encoder[level](x)
            encoder_out.append(x)
            x = self.poolers[level](x)

        # apply base
        x = self.base(x)

        # apply decoder path
        encoder_out = encoder_out[::-1]
        for level in range(self.depth):
            x = self.upsamplers[level](x)
            x = self.decoder[level](self._crop_and_concat(x,
                                                          encoder_out[level]))

        # apply output conv and activation (if given)
        x = self.out_conv(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


class UNet2d(UNetBase):
    """ 2d U-Net for segmentation as described in
    https://arxiv.org/abs/1505.04597
    """

    # Convolutional block for single layer of the decoder / encoder
    # we apply to 2d convolutions with relu activation
    def _conv_block(self, in_channels, out_channels, level, part, activation=nn.ReLU()):
        """
        Returns a 'double' conv block as described in the paper.
        Group Norm can be skipped until specified level
        :param in_channels:
        :param out_channels:
        :param level:
        :param part:
        :param activation:
        :return:
        """
        padding = 1 if self.pad_convs else 0
        if self.norm is not None:
            num_groups1 = min(in_channels, self.norm_groups)
            num_groups2 = min(out_channels, self.norm_groups)
        else:
            num_groups1 = None
            num_groups2 = None
        if self.norm is None or (self.skip_gn_level is not None and self.skip_gn_level >= level):
            sequence = nn.Sequential(nn.Conv2d(in_channels, out_channels,
                                               kernel_size=3, padding=padding),
                                     activation,
                                     nn.Conv2d(out_channels, out_channels,
                                               kernel_size=3, padding=padding),
                                     activation)
        elif self.norm == 'GroupNorm':
            sequence = nn.Sequential(nn.GroupNorm(num_groups1, in_channels),
                                     nn.Conv2d(in_channels, out_channels,
                                               kernel_size=3, padding=padding),
                                     activation,
                                     nn.GroupNorm(num_groups2, out_channels),
                                     nn.Conv2d(out_channels, out_channels,
                                               kernel_size=3, padding=padding),
                                     activation)

        if self.p_dropout is not None:
            sequence.add_module('droupout', nn.Dropout2d(p=self.p_dropout))

        return sequence

    # upsampling via transposed 2d convolutions
    def _upsampler(self, in_channels, out_channels, level, mode):
        # use bilinear upsampling + 1x1 convolutions
        return Upsample(in_channels=in_channels,
                        out_channels=out_channels,
                        scale_factor=2, mode=mode, ndim=2,
                        align_corners=False if mode == 'bilinear' else None)

    # pooling via maxpool2d
    def _pooler(self, level):
        if self.pool_mode == 'MaxPool':
            return nn.MaxPool2d(2)
        elif self.pool_mode == 'StrideConv':
            return nn.Conv2d(self.features_per_level[level], self.features_per_level[level],
                             kernel_size=2, stride=2, padding=0)

    def _out_conv(self, in_channels, out_channels):
        return nn.Conv2d(in_channels, out_channels, 1)
