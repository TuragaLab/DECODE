import datetime
import matplotlib.pyplot as plt
import numpy as np
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.autograd import Variable
from scipy import signal, ndimage, special

import evaluation as eva
import train
import psf_kernel
import simulator
from train import GaussianSmoothing, SMLMDataset, load_model


class Prediction:

    batch_size = 10
    shuffle = True
    num_workers = 4

    def __init__(self, model_file, test_data_file, transform=['normalise'],
                 extent=(32, 32), upscale_factor=1, photon_threshold=100, distance_threshold=0.5):

        self.model = None
        self.model_file = model_file
        self.test_data = None
        self.test_data_file = test_data_file
        self.extent = extent
        self.upscale_factor = upscale_factor
        self.transform = transform
        self.photon_threshold = photon_threshold
        self.distance_threshold = distance_threshold

        self.frames = []

        self.tp_cum = 0
        self.fp_cum = 0
        self.fn_cum = 0

    def load_data(self):
        self.test_data = SMLMDataset(self.test_data_file,
                                     transform=self.transform)

        self.td_loader = DataLoader(self.test_data,
                                    batch_size=Prediction.batch_size,
                                    shuffle=Prediction.shuffle,
                                    num_workers=Prediction.num_workers)

    def get_next_batch(self):
        input, output, ix = next(iter(self.td_loader))

        return input, output, ix

    def get_and_evaluate_batch(self):
        input, output, ix = self.get_next_batch()

        for i in range(ix.numel()):
            if self.test_data.emitters[ix[i]].numel() != 0:  # maybe change later
                self.frames.append(FrameEvaluation(output[i, 0, :, :], self.test_data.emitters[ix[i]],
                                                   self.photon_threshold, self.distance_threshold,
                                                   self.upscale_factor))
                self.frames[-1].run_frame_evaluation()

                self.tp_cum += self.frames[-1].tp_count
                self.fp_cum += self.frames[-1].fp_count
                self.fn_cum += self.frames[-1].fn_count

    def load_model(self):
        self.model = load_model(self.model_file)
        self.model = model.eval()


class FrameEvaluation:

    def __init__(self, output, em_mat_gt, photon_threshold, distance_threshold, upscale_factor):
        self.frame = output
        self.em_mat_gt = em_mat_gt
        self.em_mat_gt_red = None

        self.photon_threshold = photon_threshold
        self.distance_threshold = distance_threshold
        self.upscale_factor = upscale_factor

        self.em_mat_out = None
        self.photc_out = None

        self.tp_lix = None
        self.fp_lix = None
        self.fn_lix = None

        self.rmse_loss = None

    def run_frame_evaluation(self):
        # discard ground truth emitters with less than threshold photon_threshold
        self.em_mat_gt_red = self.em_mat_gt[self.em_mat_gt[:, 3]
                                            >= self.photon_threshold, :2]
        self.photc_gt_red = self.em_mat_gt[self.em_mat_gt[:, 3]
                                           >= self.photon_threshold, 3]

        self.em_mat_out, _, self.photc_out = eva.image2emitter(self.frame,
                                                               threshold=self.photon_threshold,
                                                               order='reverse')

        self.em_mat_out = eva.px_ix_2_coordinate(self.em_mat_out, scf=self.upscale_factor)

        # tp, fp, fn, ix_p2t, ix_t, ix_t2p, ix_p = iterative_pos_neg(emitter_pos_out,
        #                                                           emitter_pos_gt,
        #                                                           distance_threshold=.5)
        tp, fp, fn, ix_p2t, ix_t, ix_t2p, ix_p = \
            eva.pos_neg_emitters(self.em_mat_out,
                                 self.em_mat_gt_red,
                                 distance_threshold=self.distance_threshold)

        self.tp_lix = tp
        self.fp_lix = fp
        self.fn_lix = fn

        if (self.em_mat_out[ix_p, :].numel() != 0) and (self.em_mat_gt_red[ix_t2p, :].numel() != 0):

            # channel 0: prediction, channel 1: ground_truth
            self.pred_2_gt_pair = torch.cat((self.em_mat_out[ix_p, :].unsqueeze(0),
                                             self.em_mat_gt_red[ix_t2p, :].unsqueeze(0)),
                                            dim=0)
            self.gt_2_pred_pair = torch.cat((self.em_mat_out[ix_p2t, :].unsqueeze(0),
                                             self.em_mat_gt_red[ix_t, :].unsqueeze(0)),
                                            dim=0)

            self.rmse_loss = eva.rmse_loss(self.pred_2_gt_pair[0, :, :],
                                            self.pred_2_gt_pair[1, :, :])
        else:
            self.pred_2_gt_pair = None
            self.gt_2_pred_pair = None
            self.rmse_loss = 0.

    @property
    def tp_count(self):
        return self.tp_lix.sum()

    @property
    def fp_count(self):
        return self.fp_lix.sum()

    @property
    def fn_count(self):
        return self.fn_lix.sum()

    @property
    def tp(self):
        return self.em_mat_out[self.tp_lix, :]

    @property
    def fp(self):
        return self.em_mat_out[self.fp_lix, :]

    @property
    def fn(self):
        return self.em_mat_gt_red[self.fn_lix, :]
