import datetime
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.autograd import Variable
from scipy import signal, ndimage, special

from model import DeepSLMN
from psf_kernel import GaussianSmoothing


class SMLMDataset(Dataset):
    def __init__(self, inputfile, transform=None):
        super().__init__()

        self.transform = transform
        self.images = None
        self.images_hr = None
        self.emitters = None

        # load numpy binary
        bin = np.load(inputfile)

        # from simulator it comes in x,y,batch_ix format, we want to change to NCHW (i.e. batch_ix, channel_ix, x,y)
        img, img_hr, emitters = np.swapaxes(bin['frames'], 0, 2)[:, None, :, :], \
                                np.swapaxes(bin['frames_hr'], 0, 2)[:, None, :, :], \
                                bin['emitters']

        self.images = torch.from_numpy(img.astype(np.float32))
        self.images_hr = torch.from_numpy(img_hr.astype(np.float32))
        self.emitters = torch.from_numpy(emitters.astype(np.float32))

        # double check that we have correct number of samples
        if self.images.shape[0] != self.images_hr.shape[0]:
            raise ValueError('Dataset has unequal amount of images to ground_truth')

        # transform
        if self.transform is not None:
            if 'project01' in self.transform:
                self.images = project01(self.images)
                # self.images_hr = project01(self.images_hr)
            if 'normalise'  in self.transform:
                mean = self.images.mean()
                std = self.images.std()

                self.images = normalise(self.images, mean, std)

        print("Dataset of {} samples loaded.".format(self.__len__()))

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, index):

        return self.images[index, :, :, :], self.images_hr[index, :, :, :]


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
    # batch
    for ix, data_i in enumerate(data, 0):
        input, ground_truth = data_i

        if torch.cuda.is_available():  # model_deep.cuda():
            input, ground_truth = input.cuda(), ground_truth.cuda()
        input, target = Variable(input), Variable(ground_truth)

        opt.zero_grad()

        output = model(input)

        loss = crit(output, ground_truth)
        loss.backward()
        opt.step()

        if ix % 10 == 0:
            print(loss.data)


def test(data, model, crit):
    pass


def bump_mse_loss(output, target, kernel_pred, kernel_true=lambda x: x, l1=torch.nn.L1Loss(), l2=torch.nn.MSELoss(), l1_sc=1, l2_sc=1):
    heatmap_pred = kernel_pred(output)
    heatmap_true = kernel_true(target)

    l1_loss = l1(output, torch.zeros('like', target))
    l2_loss = l2(heatmap_pred, heatmap_true)

    return l1_sc * l1_loss + l2_sc * l2_loss  # + 10**(-2) * loss_num


def num_active_emitter_loss(input, target, threshold=0.15):
    input_f = input.view(*input.shape[:2], -1)
    target_f = target.view(*target.shape[:2], -1)

    num_true_emitters = torch.sum(target_f > threshold * target_f.max(), 2)
    num_pred_emitters = torch.sum(input_f > threshold * input_f.max(), 2)

    loss = ((num_pred_emitters - num_true_emitters) ** 2).sum() / input.__len__()
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
        last_net =  len(os.listdir(net_folder)) - 1
        file = '{}/net_{}.pt'.format(net_folder, last_net)

    if torch.cuda.is_available():
        return torch.load(file)
    else:
        return torch.load(file, map_location='cpu')


if __name__ == '__main__':
    net_folder = 'network'
    epochs = 1000

    data_smlm = SMLMDataset('data/data_32px_1e6.npz', transform=['project01', 'normalise'])
    model_deep = load_model('network/net_14.pt')
    #model_deep = DeepSLMN()
    #model_deep.weight_init()
    optimiser = Adam(model_deep.parameters(), lr=0.001)

    gaussian_kernel = GaussianSmoothing(1, [7, 7], 1, dim=2, cuda=torch.cuda.is_available(),
                                        padding=lambda x: F.pad(x, (3, 3, 3, 3), mode='reflect'))
    criterion = lambda input, target: bump_mse_loss(input, target,
                                                    kernel_pred=gaussian_kernel, kernel_true=gaussian_kernel)

    if torch.cuda.is_available():
        model_eep = model_deep.cuda()

    train_size = int(0.8 * len(data_smlm))
    test_size = len(data_smlm) - train_size
    train_data, test_data = torch.utils.data.random_split(data_smlm, [train_size, test_size])

    train_loader = DataLoader(train_data, batch_size=100, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_data, batch_size=100, shuffle=False, num_workers=4)

    for i in range(epochs):
        print('Epoch no.: {}'.format(i))
        train(train_loader, model_deep, optimiser, criterion)
        save_model(model_deep, i, filename='test.pt')
