import torch
from torch.utils.data import Dataset

from deepsmlm.generic.psf_kernel import DeltaPSF, DualDelta


class SMLMDataset(Dataset):
    """
    A SMLMDataset derived from the Dataset class.
    """
    def __init__(self, binary_loader, input_file):
        """
        Class constructor.

        :param inputfile:   binary file consistent to the format which is needed here.
        """
        super().__init__()

        self.frames = None
        self.image_shape = None
        self.em = None
        self.extent = None
        self.upsampling = 8

        em_total, extent, self.frames = binary_loader(input_file)
        self.em = em_total.split_in_frames()

        self.image_shape = tuple(self.frames.shape[2:])
        self.image_shape_hr = (self.image_shape[0] * self.upsampling,
                               self.image_shape[1] * self.upsampling)
        self.extent = (extent[0], extent[1], None)

        """Target data generation. Borrowed from psf-kernel."""
        self.target_generator = DeltaPSF(xextent=self.extent[0],
                                          yextent=self.extent[1],
                                          zextent=self.extent[2],
                                          img_shape=self.image_shape_hr)

        print("Dataset of {} samples loaded.".format(self.__len__()))

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

        if index == 0:

            img = torch.cat((
                self.frames[0, :, :, :],
                self.frames[0, :, :, :],
                self.frames[1, :, :, :]), dim=0)

        elif index == (self.__len__() - 1):

            img = torch.cat((
                self.frames[-2, :, :, :],
                self.frames[-1, :, :, :],
                self.frames[-1, :, :, :]), dim=0)

        else:

            img = torch.cat((
                self.frames[index - 1, :, :, :],
                self.frames[index, :, :, :],
                self.frames[index + 1, :, :, :]), dim=0)

        """
        Representation of the emitters on a grid, where each pixel / voxel is used for one emitter.
        """
        one_hot_img = self.target_generator.forward(self.em[index].xyz[:, :2],
                                                    self.em[index].phot)

        return img, one_hot_img
