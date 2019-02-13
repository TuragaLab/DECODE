import torch
from torch.utils.data import Dataset

from ..generic.psf_kernel import DualDelta


class SMLMDataset(Dataset):
    """
    A SMLMDataset derived from the Dataset class.
    """
    def __init__(self, input_file):
        """
        Class constructor.

        :param inputfile:   binary file consistent to the format which is needed here.
        """
        super().__init__()

        self.images = None
        self.image_shape = None
        self.em = None
        self.extent = None
        self.upsampling = 8

        self.images, self.em = load_binary(input_file)
        self.image_shape = tuple(self.images.shape[2:])
        self.image_shape_hr = (self.image_shape[0] * self.upsampling,
                               self.image_shape[1] * self.upsampling,
                               self.image_shape[2] * self.upsampling)

        self.extent = ((-0.5, self.images_shape[0] - 0.5), (-0.5, self.image_shape[1] - 0.5), None)

        """Target data generation. Borrowed from psf-kernel."""
        self.target_generator = DualDelta(xextent=self.extent[0],
                                          yextent=self.extent[1],
                                          zextent=self.extent[2],
                                          img_shape=self.image_shape_hr)

        print("Dataset of {} samples loaded.".format(self.__len__()))

    def __len__(self):
        """
        Get the length of the dataset.

        :return:    length of the dataset.
        """

        return self.images.shape[0]

    def __getitem__(self, index):
        """
        Method to retrieve a sample.

        :param index: index of the sample.
        :return: a sample, i.e. an input image and a target
        """

        if index == 0:

            img = torch.cat((
                self.images[0, :, :, :],
                self.images[0, :, :, :],
                self.images[1, :, :, :]), dim=0)

        elif index == (self.__len__() - 1):

            img = torch.cat((
                self.images[-2, :, :, :],
                self.images[-1, :, :, :],
                self.images[-1, :, :, :]), dim=0)

        else:

            img = torch.cat((
                self.images[index - 1, :, :, :],
                self.images[index, :, :, :],
                self.images[index + 1, :, :, :]), dim=0)

        """
        Representation of the emitters on a grid, where each pixel / voxel is used for one emitter.
        """
        one_hot_img = self.target_generator.forward(self.em[index].xyz[:, :2],
                                                    self.em[index].phot,
                                                    self.em[index].xyz[:, 2])

        return img, one_hot_img, index
