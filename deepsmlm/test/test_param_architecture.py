import torch
import pytest
import numpy as np
import matplotlib.pyplot as plt
from torch import nn

import deepsmlm.test.utils_ci as tutil
import deepsmlm.neuralfitter.pre_processing as prep
import deepsmlm.neuralfitter.weight_generator as wgen

import deepsmlm.neuralfitter.models.unet_parameterised as arch


class TestMUnet:

    @pytest.fixture(scope='class')
    def mu_net(self):
        return arch.MUNet(
            in_channels=3,
            n_classes=6,
            depth=3,
            wf=7,
            mlt_head='branched',
            branch_depth=2,
            branch_filters=64,
            padding=True,
            batch_norm=False,
            up_mode='upsample',
            activation=nn.ReLU(),
            last_activation=True
        )

    def test_forward(self, mu_net):
        with torch.autograd.set_detect_anomaly(True):
            x = torch.zeros((32, 3, 64, 64)).cuda()
            mu_net = mu_net.cuda()
            out = mu_net.forward(x)
            out.sum().backward()
            print("Done.")
