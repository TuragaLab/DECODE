import torch
from torch.utils.data import Dataset
from dataprep import one_hot_dual, load_binary


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

        self.transform = transform
        self.images = None
        self.image_shape = None
        self.em = None
        self.extent = None
        self.upsampling = 8

        self.images, self.em = load_binary(input_file)
        self.image_shape = torch.tensor(self.images.shape[2:])
        self.extent = ((-0.5, self.images_shape[0] - 0.5), (-0.5, self.image_shape[1] - 0.5))

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
        # This is a psf method!
        one_hot_img = one_hot_dual(self.em[index],
                                   self.image_shape,
                                   self.upsampling,
                                   self.extent[0],
                                   self.extent[1])

        return img, one_hot_img, index
