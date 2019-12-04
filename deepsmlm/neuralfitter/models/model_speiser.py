import torch
import torch.nn as nn
import torch.nn.functional as F

import math


def icnr(x, scale=2, init=nn.init.xavier_uniform_):
    "ICNR init of `x`, with `scale` and `init` function. Adopted from the Theano implementation"
    shape = x.shape

    stride = (scale, len(shape) - 2)
    subshape = shape[:2] + tuple(l // s for l, s in zip(shape[2:], stride))
    result = init(torch.zeros(subshape))
    for d, s in enumerate(stride):
        result = torch.repeat_interleave(result, s, 2 + d)
    result = result[(slice(None), slice(None)) + tuple(slice(None, l) for l in shape[2:])]
    x.data.copy_(result)


class SUNet(nn.Module):
    def __init__(self, n_inp, n_filters=64, n_stages=2):
        super(SUNet, self).__init__()
        curr_N = n_filters
        self.n_stages = n_stages
        self.layer_path = nn.ModuleList()

        self.layer_path.append(nn.Conv2d(n_inp, curr_N, kernel_size=3, padding=1))
        self.layer_path.append(nn.Conv2d(curr_N, curr_N, kernel_size=3, padding=1))

        for i in range(n_stages):
            self.layer_path.append(nn.Conv2d(curr_N, curr_N, kernel_size=2, stride=2, padding=0))
            self.layer_path.append(nn.Conv2d(curr_N, curr_N * 2, kernel_size=3, padding=1))
            curr_N *= 2
            self.layer_path.append(nn.Conv2d(curr_N, curr_N, kernel_size=3, padding=1))

        for i in range(n_stages):
            self.layer_path.append(nn.ConvTranspose2d(curr_N, curr_N // 2, kernel_size=2, stride=2))
            icnr(self.layer_path[-1].weight)
            self.layer_path[-1].is_icnr = True

            curr_N = curr_N // 2

            self.layer_path.append(nn.Conv2d(curr_N * 2, curr_N, kernel_size=3, padding=1))
            self.layer_path.append(nn.Conv2d(curr_N, curr_N, kernel_size=3, padding=1))

        for m in self.layer_path:
            if isinstance(m, nn.Conv2d):
                if not hasattr(m, 'is_icnr'):
                    nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')

    def forward(self, x):

        n_l = 0
        x_bridged = []

        x = F.elu(list(self.layer_path)[n_l](x));
        n_l += 1;
        x = F.elu(list(self.layer_path)[n_l](x));
        n_l += 1;
        x_bridged.append(x)

        for i in range(self.n_stages):
            for n in range(3):
                x = F.elu(list(self.layer_path)[n_l](x));
                n_l += 1;
                if n == 2 and i < self.n_stages - 1:
                    x_bridged.append(x)

        for i in range(self.n_stages):
            for n in range(3):
                x = F.elu(list(self.layer_path)[n_l](x));
                n_l += 1;
                if n == 0:
                    x = torch.cat([x, x_bridged.pop()], 1)
        return x


if __name__ == '__main__':
    model = SUNet(3, 48)
    x = torch.rand((2, 3, 32, 32))
    out = model.forward(x)
    out.sum().backward()

    print("Done.")
