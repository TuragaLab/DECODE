import torch
from torch.utils.data import Dataset

from deepsmlm.generic.psf_kernel import DeltaPSF, DualDelta, ListPseudoPSF
from deepsmlm.neuralfitter.pre_processing import RemoveOutOfFOV


class SMLMDataset(Dataset):
    """
    A SMLMDataset derived from the Dataset class.
    """
    def __init__(self, emitter, extent, frames, multi_frame_output=True, dimensionality=3):
        """

        :param emitter: set of emitters loaded by binary loader
        :param extent: extent of the dataset
        """

        super().__init__()

        self.frames = frames
        self.image_shape = None
        self.em = None
        self.extent = (extent[0], extent[1], None)
        self.upsampling = 8
        self.multi_frame_output = multi_frame_output
        self.dimensionality = dimensionality

        # Remove the emitters which are out of the FOV.
        emitter = RemoveOutOfFOV(self.extent[0], self.extent[1]).clean_emitter_set(emitter)
        self.em = emitter.split_in_frames(ix_f=0, ix_l=self.__len__()-1)

        self.image_shape = tuple(self.frames.shape[2:])
        self.image_shape_hr = (self.image_shape[0] * self.upsampling,
                               self.image_shape[1] * self.upsampling)

        """Target data generation. Borrowed from psf-kernel."""
        self.target_generator = ListPseudoPSF(xextent=self.extent[0],
                                              yextent=self.extent[1],
                                              zextent=None,
                                              zero_fill_to_size=20,
                                              dim=self.dimensionality)
        # self.target_generator = DeltaPSF(xextent=self.extent[0],
        #                                  yextent=self.extent[1],
        #                                  zextent=None,
        #                                  img_shape=self.image_shape_hr)

        print("Dataset loaded. N: {} samples.".format(self.__len__()))

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

        """Get adjacent frames. Pad borders with 'same'. Therefore we use the max(0, ix-1) and min(lastix, index+1)."""
        if self.multi_frame_output:
            img = torch.cat((
                self.frames[max(0, index - 1), :, :, :],
                self.frames[index, :, :, :],
                self.frames[min(self.__len__() - 1, index + 1), :, :, :]), dim=0)
        else:
            img = self.frames[index, :, :, :]

        # """
        # Representation of the emitters on a grid, where each pixel / voxel is used for one emitter.
        # """
        """List of Emitters. 2D"""
        pos_tar, phot_tar = self.target_generator.forward(self.em[index].xyz, self.em[index].phot)
        target = (pos_tar, phot_tar)
        return img, target, index


class SMLMDatasetOnFly(Dataset):
    def __init__(self, extent, prior, simulator, data_set_size, dimensionality=3, reuse=False):
        super().__init__()

        self.extent = extent
        self.dimensionality = dimensionality
        self.data_set_size = data_set_size
        self.reuse = reuse

        self.prior = prior
        self.simulator = simulator

        self.target_generator = ListPseudoPSF(xextent=self.extent[0],
                                              yextent=self.extent[1],
                                              zextent=self.extent[2],
                                              zero_fill_to_size=64,
                                              dim=self.dimensionality)

        """Pre-Calculcate the complete dataset and use the same data as one draws samples. 
        This is useful for the testset."""
        if self.reuse:
            self.frame = [None] * self.__len__()
            self.target = [None] * self.__len__()

            for i in range(self.__len__()):
                emitter = self.prior.pop()
                self.frame[i] = self.simulator.background.forward(
                    self.simulator.psf.forward(emitter.xyz, emitter.phot)).type(torch.FloatTensor)
                self.target[i] = self.target_generator.forward(emitter.xyz, emitter.phot)

        else:
            self.frame = None
            self.target = None

    def __len__(self):
        return self.data_set_size

    def __getitem__(self, index):

        if not self.reuse:
            emitter = self.prior.pop()
            frame = self.simulator.background.forward(self.simulator.psf.forward(emitter.xyz, emitter.phot)).type(torch.FloatTensor)
            target = self.target_generator.forward(emitter.xyz, emitter.phot)
        else:
            frame = self.frame[index]
            target = self.target[index]

        return frame, target, index


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
