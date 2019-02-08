import datetime
import numpy as np
import scipy
import os
import sys
import functools
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.autograd import Variable

from psf_kernel import delta_psf


def load_binary(bin_path):

    if bin_path[-3:] in ('.np', 'npz'):  # load numpy binary
        bin = np.load(bin_path)

    elif bin_path[-4:] == '.mat':
        bin = scipy.io.loadmat(bin_path)
    else:
        raise ValueError('Datatype not supported.')

    frames = np.ascontiguousarray(bin['frames'], dtype=np.float32)
    em_mat = np.ascontiguousarray(bin['emitters'], dtype=np.float32)

    return torch.from_numpy(frames), torch.from_numpy(em_mat)


def generate_3d_hr_target_frame(em_mat, framedim, upsampling, xextent, yextent):
    hr_img_dim = framedim * upsampling
    num_channels = 2  # namely photon count and z position

    target = torch.zeros((num_channels, hr_img_dim[0], hr_img_dim[1]))
    target[[0], :, :] = delta_psf(em_mat[:, :2], em_mat[:, 3], img_shape=hr_img_dim, xextent=xextent, yextent=yextent)
    target[[1], :, :] = delta_psf(em_mat[:, :2], em_mat[:, 2], img_shape=hr_img_dim, xextent=xextent, yextent=yextent)

    return target


def generate_3d_hr_target(em_mat, framedim, upsampling, xextent, yextent):
    num_frames = int(em_mat[:, 4].max() + 1)
    hr_img_dim = framedim * upsampling

    num_channels = 2  # namely photon count and z position

    target = torch.zeros((num_frames, num_channels, hr_img_dim[0], hr_img_dim[1]))

    for f in range(num_frames):
        ix_in_frame = em_mat[:, 4] == f
        xy = em_mat[ix_in_frame, :2]
        z =  em_mat[ix_in_frame, 2]
        phot =  em_mat[ix_in_frame, 3]

        target[f, [0], :, :] = delta_psf(xy, phot, img_shape=hr_img_dim, xextent=xextent, yextent=yextent)
        target[f, [1], :, :] = delta_psf(xy, z, img_shape=hr_img_dim, xextent=xextent, yextent=yextent)
    return target


if __name__ == '__main__':
    frames, em_mat = load_binary('data/spline_1e4.mat')
    target = generate_3d_hr_target(em_mat, np.array([32, 32]), 8, np.array([-0.5, 31.5]), np.array([-0.5, 31.5]))

    plt.figure()
    plt.subplot(121)
    plt.imshow(target[0, 0, :, :])
    plt.subplot(122)
    plt.imshow(target[0, 1, :, :])
    plt.show()