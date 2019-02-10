import datetime
import importlib
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import torch.tensor as tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, SGD
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.autograd import Variable

import psf_kernel
import simulator


def mle_objective(output, target, t=0):
    chisq = (output - target).sum() - \
            (target[target > t] * torch.log(output[target > t] / target[target > t])).sum()
    return chisq * 2


def pos2grid(pos, px_extent):
    return pos.floor() + ((pos - pos.floor()) / px_extent).ceil() * px_extent


def sim_uncertainty(pos=None, pos_uc=None, phot=None, phot_uc=None, bg=None, bg_uc=None):
    if pos is not None:
        pos = pos + torch.randn_like(pos) * pos_uc
    if phot is not None:
        phot = (phot * (1 + torch.randn_like(phot) * phot_uc))
    if bg is not None:
        bg = bg + torch.randn_like(bg) * bg_uc

    return pos, phot, bg


def fitter(psf, optimiser, input_image, xy_grad, phot_grad, xy_fix, phot_fix, iterations):
    for i in range(100):
        optimiser.zero_grad()

        xy = torch.cat((xy_2fit, xy_fix), 0)
        phot = torch.cat((phot_2fit, phot_fix), 0)
        out_image = psf_kernel.gaussian_expect(xy, sigma, phot, img_shape, shotnoise)

        l = loss(out_image, input_image)
        l.backward()
        optimiser.step()

    return xy, phot


def dummy_simulation(num_frames, avg_emitter_p_frame=3, z_sigma=3):
    em_mat, cont_mat = simulator.random_emitters(avg_emitter_p_frame, num_frames, None, (16, 16), z_sigma=z_sigma)

    xyz, phot, _ = sim_uncertainty(em_mat[:, 0:3], pos_uc=4 / 8, phot=em_mat[:, 3], phot_uc=0.15)
    xyz = pos2grid(xyz, 1 / 8)
    em_mat_init = torch.cat((xyz, phot.unsqueeze(1), em_mat[:, 4:]), 1)

    xyz, phot, _ = sim_uncertainty(cont_mat[:, 0:3], pos_uc=4 / 8, phot=cont_mat[:, 3], phot_uc=0.15)
    xyz = pos2grid(xyz, 1 / 8)
    cont_mat_init = torch.cat((xyz, phot.unsqueeze(1), cont_mat[:, 4:]), 1)

    return em_mat, cont_mat, em_mat_init, cont_mat_init


if __name__ == '__main__':
    emit, cont, em_ini, cont_ini = dummy_simulation(num_frames=10, avg_emitter_p_frame=5, z_sigma=0)
    args = simulator.Args()
    args.config['DirectConfiguration'] = {'binary_path': 'data/temp.npz',
                                          'positions_csv': '',
                                          'image_size': '(16, 16)',
                                          'upscale_factor': '1',
                                          'emitter_p_frame': '0',
                                          'total_frames': '0',
                                          'bg_value': '10',
                                          'sigma': '1.5',
                                          'poolsize': '10',
                                          'use_cuda': 'True',
                                          'dimension': '3'}
    args.parse(section='DirectConfiguration', from_variable=True)

    sim = simulator.Simulation(emit, cont,
                               img_size=args.image_size,
                               sigma=args.sigma,
                               upscale=args.upscale_factor,
                               bg_value=args.bg_value,
                               background=None,
                               psf=None,
                               psf_hr=None,
                               poolsize=args.poolsize,
                               use_cuda=args.use_cuda)

    sim.run_simulation(plot_sample=False)

    shotnoise = False
    sigma = args.sigma
    img_shape = args.image_size

    loss = mle_objective

    for f in range(sim.num_frames):

        cont_f = cont_ini[cont_ini[:, 4] == f, :]  # sim.get_emitter_matrix_frame(f, kind='contaminator')
        emit_f = em_ini[em_ini[:, 4] == f, :]

        if emit_f.numel() == 0:
            continue

        xyz_2fit = emit_f[:, :3].clone().detach().requires_grad_(True).type(torch.float32)
        phot_2fit = emit_f[:, 3].clone().detach().requires_grad_(True).type(torch.float32)
        bg_2fit = torch.tensor([5.], requires_grad=True, dtype=torch.float32)  # common term

        xyz_fix = cont_f[:, :3].clone().detach().requires_grad_(False).type(torch.float32)
        phot_fix = cont_f[:, 3].clone().detach().requires_grad_(False).type(torch.float32)

        xyz_gt = sim.get_emitter_matrix_frame(f, kind='emitter')[:, :3]
        phot_gt = sim.get_emitter_matrix_frame(f, kind='emitter')[:, 3]

        raw_image = sim.image[f, :, :, :].type(torch.float32)
        optimiser = Adam([
            {'params': [xyz_2fit], 'lr': 1e-1},
            {'params': [phot_2fit], 'lr': 10},
            {'params': [bg_2fit], 'lr': 1}], lr=0.001)

        for i in range(100):
            optimiser.zero_grad()

            xyz = torch.cat((xyz_2fit, xyz_fix), 0)
            phot = torch.cat((phot_2fit, phot_fix), 0)
            out_image = psf_kernel.gaussian_expect(xyz, sigma, phot, img_shape, shotnoise, bg=bg_2fit)

            l = loss(out_image, raw_image)
            l.backward()
            optimiser.step()

        plt.figure(figsize=(16, 16))
        plt.subplot(121, title='Input image')
        plt.imshow(raw_image[0, :, :].detach(), cmap='gray')
        plt.scatter(xyz_gt[:, 0], xyz_gt[:, 1], marker='x')
        print('Target: xy-photons\n{}\nBg: {}\n'.format(torch.cat((xyz_gt, phot_gt.unsqueeze(1)), 1), args.bg_value))

        plt.subplot(122, title='Modeled image')
        plt.imshow(out_image[0, :, :].detach(), cmap='gray')
        plt.scatter(emit_f[:, 0].detach(), emit_f[:, 1].detach(), marker='+', c='g')
        plt.scatter(xyz_gt[:, 0], xyz_gt[:, 1], marker='x', c='r')
        plt.scatter(xyz_2fit[:, 0].detach(), xyz_2fit[:, 1].detach(), marker='+', c='b')
        print('Output: xy-photons\n{}\nBg: {}'.format(torch.cat((xyz_2fit, phot_2fit.unsqueeze(1)), 1), float(bg_2fit)))
        plt.pause(0.001)