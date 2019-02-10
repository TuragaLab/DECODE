import functools
import numpy as np
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.autograd import Variable
from scipy import signal, ndimage, special

import dataprep
from simulator import upscale
from model import DeepSLMN
from psf_kernel import GaussianSmoothing


def train(data, model, opt, crit):

    model.train()

    print_steps = torch.round(torch.linspace(0, data.__len__(), 5))

    for ix, data_i in enumerate(data, 0):
        input, ground_truth, _ = data_i

        if torch.cuda.is_available():  # model_deep.cuda():
            input, ground_truth = input.cuda(), ground_truth.cuda()
        input, target = Variable(input), Variable(ground_truth)

        opt.zero_grad()

        output = model(input)

        loss = crit(output, ground_truth)
        loss.backward()
        opt.step()

        if ix in print_steps:
            print(loss.data)


def test(data, model, crit):
    pass


def save_model(model, epoch, net_folder='network', filename=None):

    if filename is None:
        if epoch == 0:
            file_ix = len(os.listdir(net_folder))
            torch.save(model, '{}/net_{}.pt'.format(net_folder, file_ix))
        else:
            file_ix = len(os.listdir(net_folder)) - 1
            torch.save(model, '{}/net_{}.pt'.format(net_folder, file_ix))
    else:
        torch.save(model, '{}/{}'.format(net_folder, filename))


def load_model(file=None, net_folder='network'):
    if file is None:
        last_net = len(os.listdir(net_folder)) - 1
        file = '{}/net_{}.pt'.format(net_folder, last_net)

    if torch.cuda.is_available():
        return torch.load(file)
    else:
        return torch.load(file, map_location='cpu')


if __name__ == '__main__':
    if len(sys.argv) == 1:  # no .ini file specified
        dataset_file = 'data/spline_1e5.mat'
        weight_in = 'network/spline_1e4_no_z.pt'
        weight_out = 'spline_1e5_no_z.pt'
    else:
        dataset_file = sys.argv[1]
        weight_in = None if sys.argv[2].__len__() == 0 else sys.argv[2]
        weight_out = sys.argv[3]

    net_folder = 'network'
    epochs = 1000
    if weight_in is None:
        model_deep = DeepSLMN()
        model_deep.weight_init()
    else:
        model_deep = load_model(weight_in)

    data_smlm = SMLMDataset(dataset_file, transform=None)

    optimiser = Adam(model_deep.parameters(), lr=0.001)

    gaussian_kernel = GaussianSmoothing(1, [7, 7], 1, dim=2, cuda=torch.cuda.is_available(),
                                        padding=lambda x: F.pad(x, [3, 3, 3, 3], mode='reflect'))

    def criterion(input, target): return bump_mse_loss_3d(input, target,
                                                          kernel_pred=gaussian_kernel,
                                                          kernel_true=gaussian_kernel,
                                                          lz_sc=0)

    if torch.cuda.is_available():
        model_eep = model_deep.cuda()

    train_size = int(0.8 * len(data_smlm))
    test_size = len(data_smlm) - train_size
    #train_data, test_data = torch.utils.data.random_split(data_smlm, [train_size, test_size])

    train_loader = DataLoader(data_smlm, batch_size=32, shuffle=True, num_workers=12)
    # test_loader = DataLoader(test_data, batch_size=32, shuffle=False, num_workers=12)

    for i in range(epochs):
        print('Epoch no.: {}'.format(i))
        train(train_loader, model_deep, optimiser, criterion)
        save_model(model_deep, i, filename=weight_out)
