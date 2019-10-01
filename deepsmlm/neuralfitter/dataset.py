import torch
import os
import glob
import time
from torch.utils.data import Dataset
import ctypes
import numpy as np
import multiprocessing as mp

from deepsmlm.generic.emitter import EmitterSet
from deepsmlm.neuralfitter.pre_processing import RemoveOutOfFOV, N2C, Identity


class SMLMDataset(Dataset):
    """
    A SMLMDataset derived from the Dataset class.
    """
    def __init__(self, emitter, extent, frames, tar_gen, multi_frame_output=True, dimensionality=3):
        """

        :param emitter: set of emitters loaded by binary loader
        :param extent: extent of the dataset
        """

        super().__init__()

        self.frames = frames
        self.image_shape = None
        self.em = None
        self.extent = extent
        self.upsampling = 1
        self.multi_frame_output = multi_frame_output
        self.dimensionality = dimensionality

        # Remove the emitters which are out of the FOV.
        emitter = RemoveOutOfFOV(self.extent[0], self.extent[1]).clean_emitter_set(emitter)
        self.em = emitter.split_in_frames(ix_f=0, ix_l=self.__len__()-1)

        self.image_shape = tuple(self.frames.shape[2:])
        self.image_shape_hr = (self.image_shape[0] * self.upsampling,
                               self.image_shape[1] * self.upsampling)

        """Target data generation. Borrowed from psf-kernel."""
        self.target_generator = tar_gen

        print("Dataset loaded. N: {} samples.".format(self.__len__()))

    def __len__(self):
        """
        :return:    length of the dataset.
        """
        return self.frames.shape[0]

    def __getitem__(self, index):
        """
        :param index: index of the sample.
        :return: a sample, i.e. an input image and a target
        """
        """Get adjacent frames. Pad borders with 'same'. Therefore we use the max(0, ix-1) and min(lastix, index+1)."""
        if self.multi_frame_output:
            img = torch.cat((
                self.frames[max(0, index - 1), :, :, :],
                self.frames[index, :, :, :],
                self.frames[min(self.__len__() - 1, index + 1), :, :, :]), dim=0)
        else:
            img = self.frames[index, :, :, :]

        """Forward Emitters thorugh target generator."""
        em_tar = self.em[index]
        target = self.target_generator.forward(em_tar)
        return img, target, em_tar, index


class SMLMUnifiedDatasetLoader(Dataset):
    """
    Load a dataset which was calculated for multiple experiments
    """
    def __init__(self, folder):
        self.folder = folder
        self.ix = -float('inf')

        self.samples = None
        self.target = None

        self.max_wait = 1800

        self.step()

    def _get_highest_ix_file(self):
        """
        Get the highest index in the folder
        :return:
        """
        dataset_files = glob.glob(self.folder + '*.pt')
        if dataset_files == []:
            return None, None

        indices = []
        for ds in dataset_files:
            fname_nofolder = ds.split('/')[-1]
            indices.append(int(fname_nofolder.partition('_')[0]))
        indices = torch.tensor(indices)
        ix = indices.max(0)[1].item()
        return ix, dataset_files[ix]

    def step(self):
        """
        Check whether we got a new dataset, if yes load it.
        :return:
        """
        ix, fname = self._get_highest_ix_file()
        """If Dataset is not yet ready, sleep."""
        time_waited = 0
        while (ix is None) and time_waited < self.max_wait:
            time.sleep(1)
            time_waited += 1
            ix, fname = self._get_highest_ix_file()

        if ix > self.ix:
            (samples, target) = torch.load(fname)
            self.samples = samples
            self.target = target
            self.ix = ix

    def __len__(self):
        if self.samples is not None:
            return self.samples.size(0)
        else:
            return 0

    def __getitem(self, index):
        """
        Get a sample.
        :param index:
        :return:
        """
        return self.samples[index], self.target[index]


class SMLMDatasetOnFly(Dataset):
    def __init__(self, extent, prior, simulator, data_set_size, in_prep, tar_gen, dimensionality=3, static=False,
                 lifetime=1, return_em_tar=False, disk_cache=False):
        """

        :param extent:
        :param prior:
        :param simulator:
        :param data_set_size:
        :param in_prep: Prepare input to NN. Any instance with forwrard method
        :param tar_gen: Generate target for learning.
        :param dimensionality:
        :param static:
        :param lifetime:
        :param return_em_tar: __getitem__ method returns em_target
        """
        super().__init__()

        self.extent = extent
        self.dimensionality = dimensionality
        self.data_set_size = data_set_size
        self.static_data = static
        self.lifetime = lifetime
        self.time_til_death = lifetime
        self.return_em_tar = return_em_tar

        self.calc_new_flag = True if (not static) else False

        self.prior = prior
        self.simulator = simulator

        self.input_preperator = in_prep  # N2C()
        self.target_generator = tar_gen

        """Initialise Frame and Target. Call drop method to create list."""
        _, frame_dummy, target_dummy, _ = self.pop_new()

        frames_base = mp.Array(ctypes.c_float, self.__len__() * frame_dummy.numel())
        frames = np.ctypeslib.as_array(frames_base.get_obj())
        frames = frames.reshape(self.__len__(), frame_dummy.size(0), frame_dummy.size(1), frame_dummy.size(2))

        target_base = mp.Array(ctypes.c_float, self.__len__() * target_dummy.numel())
        target = np.ctypeslib.as_array(target_base.get_obj())
        target = target.reshape(self.__len__(), target_dummy.size(0), target_dummy.size(1), target_dummy.size(2))

        self.frame = torch.from_numpy(frames)
        self.target = torch.from_numpy(target)
        self.em_tar = [None] * self.__len__()
        self.use_cache = False

        self.drop_data_set(verbose=False)

        """Pre-Calculcate the complete dataset and use the same data as one draws samples.
        This is useful for the testset or the classical deep learning feeling of not limited training data."""
        if self.static_data:
            for i in range(self.__len__()):
                _, frame, target, em_tar = self.pop_new()
                self.frame[i] = frame
                self.target[i] = target
                self.em_tar[i] = em_tar

            self.use_cache = True
            print("Pre-calculation of dataset done.")

    def get_gt_emitter(self, output_format='list'):
        """
        Get the complete ground truth. Should only be used for static data.
        :param output_format: either list (list of emittersets) or concatenated Emittersets.
        :return:
        """
        if not self.static_data:
            print("WARNING: Ground truth extraction may not be valid for non-static data. Please be aware.")

        if output_format == 'list':
            return self.em_tar
        elif output_format == 'cat':
            return EmitterSet.cat_emittersets(self.em_tar, step_frame_ix=1)

    def step(self):
        """
        Reduce lifetime of dataset by one unit.
        :return:
        """
        self.time_til_death -= 1
        if self.time_til_death <= 0:
            self.drop_data_set()
        else:
            self.use_cache = True

    def drop_data_set(self, verbose=True):
        """
        Invalidate / clear cache.
        :param verbose: print when dropped.
        :return:
        """
        self.frame *= float('nan')
        self.target *= float('nan')
        self.em_tar = [None] * self.__len__()

        self.use_cache = False
        self.time_til_death = self.lifetime

        if verbose:
            print("Dataset dropped. Will calculate a new one in next epoch.")

    def pop_new(self):
        """

        :return: emitter (all three frames)
                 frames
                 target frames
                 emitters on the target frame (i.e. the middle frame)
        """
        emitter = self.prior.pop()
        sim_out = self.simulator.forward(emitter).type(torch.FloatTensor)
        frame = self.input_preperator.forward(sim_out)
        emitter_on_tar_frame = emitter.get_subset_frame(0, 0)
        if self.simulator.out_bg:
            target = self.target_generator.forward(emitter_on_tar_frame, sim_out[1])
        else:
            target = self.target_generator.forward(emitter_on_tar_frame)
        return emitter, frame, target, emitter_on_tar_frame

    def __len__(self):
        return self.data_set_size

    def __getitem__(self, index):

        if not self.use_cache:
            emitter, frame, target, em_tar = self.pop_new()
            self.frame[index] = frame
            self.target[index] = target
            self.em_tar[index] = em_tar

        frame, target, em_tar = self.frame[index], self.target[index], self.em_tar[index]

        if self.return_em_tar:
            return frame, target, em_tar
        else:
            return frame, target


class UnsupervisedDataset(Dataset):
    def __init__(self, extent, frames, multi_frame_output=True):
        super().__init__()

        self.frames = frames
        self.image_shape = None
        self.multi_frame_output = multi_frame_output

        self.extent = (extent[0], extent[1], None)
        self.image_shape = tuple(self.frames.shape[2:])

        print("Dataset initialised. N: {} samples.".format(self.__len__()))

    def __len__(self):
        """
        Get the length of the dataset.

        :return:    length of the dataset.
        """

        return self.frames.shape[0]

    def __getitem__(self, index):
        """
        Method to retrieve a sample.

        :param index: index of the sample.
        :return: a sample, i.e. an input image and a target
        """
        if self.multi_frame_output:
            """Get adjacent frames. Pad borders with 'same'. Therefore we use the max(0, ix-1) and min(lastix, index+1)."""
            img = torch.cat((
                self.frames[max(0, index - 1), :, :, :],
                self.frames[index, :, :, :],
                self.frames[min(self.__len__() - 1, index + 1), :, :, :]), dim=0)
        else:
            img = self.frames[index, :, :, :]

        return img, index