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


photon_count_in_px_max = 1000


class SMLMDataset(Dataset):
    def __init__(self, inputfile, transform=None, transform_vars=None):
        super().__init__()

        self.transform = transform
        self.images = None
        self.image_shape = None
        self.images_hr = None
        self.emitters = None
        self.extent = None
        self.upsampling = 8

        self.images, emitters = dataprep.load_binary(inputfile)
        self.image_shape = torch.tensor(self.images.shape[2:])

        self.extent = np.array(
            [[0, self.images.shape[2]], [0, self.images.shape[3]]])

        # self.images_hr = dataprep.generate_3d_hr_target(emitters, self.image_shape,
        #                                                 self.upsampling, self.extent[0, :], self.extent[1, :])

        # transform
        if self.transform is not None:
            if 'project01' in self.transform:
                self.images = project01(self.images)
                # self.images_hr = project01(self.images_hr)
            if 'normalise' in self.transform:
                mean = self.images.mean()
                std = self.images.std()

                torch.save([mean, std], inputfile[:-3] + '_normalisation.pt')

                self.images = normalise(self.images, mean, std)

            if 'test_set_norm_from_train' in self.transform:
                [mean, std] = torch.load(transform_vars)
                self.images = normalise(self.images, mean, std)

        #  emitter matrix as list
        self.emitters = [None] * self.__len__()
        for j in range(self.__len__()):
            self.emitters[j] = emitters[emitters[:, 4] == j, :]

        print("Dataset of {} samples loaded.".format(self.__len__()))

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, index):
        # return 3-fold images as channels. Pad images at the borders by same image
        if index == 0:
            # img[0, :, :, :].dim == 3, so channel is first dimension
            img = torch.cat(
                (self.images[0, :, :, :], self.images[0, :, :, :], self.images[1, :, :, :]), dim=0)

        elif index == (self.__len__() - 1):
            l_ix = self.__len__() - 1
            img = torch.cat(
                (self.images[l_ix - 1, :, :, :], self.images[l_ix, :, :, :], self.images[l_ix, :, :, :]), dim=0)

        else:
            img = torch.cat((self.images[index - 1, :, :, :], self.images[index,
                                                                          :, :, :], self.images[index + 1, :, :, :]), dim=0)


        # scale up and generate target
        img = upscale(img, scale=self.upsampling, img_dims=(1, 2))
        img_hr = dataprep.generate_3d_hr_target_frame(self.emitters[index], self.image_shape,
                                                      self.upsampling, self.extent[0, :], self.extent[1, :])
        return img, img_hr, index


def project01(img):
    # 4d
    img = img.contiguous()
    img_flat = img.view(img.shape[0], img.shape[1], -1)
    img_min = img_flat.min(2, keepdim=True)[0]
    img_max = img_flat.max(2, keepdim=True)[0]

    img_flat_norm = (img_flat - img_min) / (img_max - img_min)
    return img_flat_norm.view(img.shape[0], img.shape[1], img.shape[2], img.shape[3])


def normalise(img, _mean, _std):
    return (img - _mean) / _std


def get_outputsize(input_size, model):
    input = torch.randn(1, 1, input_size, input_size)
    return model.forward(input).size()


def get_gaussian_kernel(sigma=(1.5, 1.5)):
    kernel = np.outer(signal.gaussian(np.ceil(10 * sigma[0]), sigma[0]),
                      signal.gaussian(np.ceil(10 * sigma[1]), sigma[1]))
    return kernel / np.sum(kernel)


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


def bump_mse_loss_3d(output, target, kernel_pred, kernel_true, l2=torch.nn.MSELoss(), lz_sc=0.001):

    # call bump_mse_loss on first channel (photons)
    loss_photons = bump_mse_loss(output[:, [0], :, :], target[:, [0], :, :], kernel_pred, kernel_true, l1_sc=1, l2_sc=1)

    output_local_nz = kernel_pred(output[:, [0], :, :]) * kernel_pred(output[:, [1], :, :])
    target_local_nz = kernel_pred(target[:, [0], :, :]) * kernel_pred(target[:, [1], :, :])

    loss_z = l2(output_local_nz, target_local_nz)

    return loss_photons + lz_sc * loss_z


def bump_mse_loss(output, target, kernel_pred, kernel_true=lambda x: x, l1=torch.nn.L1Loss(), l2=torch.nn.MSELoss(), l1_sc=1, l2_sc=1):
    heatmap_pred = kernel_pred(output)
    heatmap_true = kernel_true(target)

    l1_loss = l1(output, torch.zeros_like(target))
    l2_loss = l2(heatmap_pred, heatmap_true)

    return l1_sc * l1_loss + l2_sc * l2_loss  # + 10**(-2) * loss_num


def interpoint_loss(input, target, threshold=500):
    def expanded_pairwise_distances(x, y=None):
        '''
        Taken from https://discuss.pytorch.org/t/efficient-distance-matrix-computation/9065

        Input: x is a Nxd matrix
               y is an optional Mxd matirx
        Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
                if y is not given then use 'y=x'.
        i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
        '''
        if y is not None:
            differences = x.unsqueeze(1) - y.unsqueeze(0)
        else:
            differences = x.unsqueeze(1) - x.unsqueeze(0)
        distances = torch.sum(differences * differences, -1)
        return distances

    interpoint_dist = expanded_pairwise_distances(
        (input >= threshold).nonzero(), target.nonzero())
    # return distance to closest target point
    return interpoint_dist.min(1)[0].sum() / input.__len__()


def inverse_intens(output, target):
    pass


def num_active_emitter_loss(input, target, threshold=0.15):
    input_f = input.view(*input.shape[:2], -1)
    target_f = target.view(*target.shape[:2], -1)

    num_true_emitters = torch.sum(target_f > threshold * target_f.max(), 2)
    num_pred_emitters = torch.sum(input_f > threshold * input_f.max(), 2)

    loss = ((num_pred_emitters - num_true_emitters)
            ** 2).sum() / input.__len__()
    return loss.type(torch.FloatTensor)


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
