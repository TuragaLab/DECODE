import torch
from torch.utils.data import Dataset

from deepsmlm.generic.psf_kernel import DeltaPSF, DualDelta, ListPseudoPSF, ListPseudoPSFInSize
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


class SMLMDatasetOnFly(Dataset):
    def __init__(self, extent, prior, simulator, data_set_size, in_prep, tar_gen, dimensionality=3, static=False,
                 lifetime=1, return_em_tar=False):
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
        self.frame = None
        self.target = None
        self.em_tar = None
        self.data_complete = None

        self.drop_data_set(verbose=False)
        """Pre-Calculcate the complete dataset and use the same data as one draws samples.
        This is useful for the testset or the classical deep learning feeling of not limited training data."""
        if self.static_data:
            for i in range(self.__len__()):
                _, frame, target, em_tar = self.pop_new()
                self.frame[i] = frame
                self.target[i] = target
                self.em_tar[i] = em_tar

            self.check_completeness(warning=True)
            self.calc_new_flag = False
            print("Pre-calculation of dataset done.")

    def step(self):
        self.time_til_death -= 1
        if self.time_til_death <= 0:
            self.drop_data_set()
            self.time_til_death = self.lifetime

    def check_completeness(self, warning=False):
        frame_complete = not any(v is None for v in self.frame)
        target_complete = not any(v is None for v in self.target)
        em_tar_complete = not any(v is None for v in self.em_tar)
        if frame_complete and target_complete and em_tar_complete:
            self.data_complete = True
            self.calc_new_flag = False
        else:
            self.data_complete = False
            self.calc_new_flag = True
            if warning:
                print("WARNING: The dataset is not complete.")

    def drop_data_set(self, verbose=True):
        self.frame = [None] * self.__len__()
        self.target = [None] * self.__len__()
        self.em_tar = [None] * self.__len__()
        self.check_completeness()

        if verbose:
            print("Dataset dropped. Will calculate a new one in next epoch.")

    def pop_new(self):
        emitter = self.prior.pop()
        sim_out = self.simulator.forward(emitter).type(torch.FloatTensor)
        frame = self.input_preperator.forward(sim_out)
        emitter_on_tar_frame = emitter.get_subset_frame(0, 0)
        target = self.target_generator.forward(emitter_on_tar_frame)
        return emitter, frame, target, emitter_on_tar_frame

    def __len__(self):
        return self.data_set_size

    def __getitem__(self, index):

        if self.calc_new_flag or self.lifetime == 0:
            emitter, frame, target, em_tar = self.pop_new()
            self.frame[index] = frame
            self.target[index] = target
            self.em_tar[index] = em_tar
        else:
            frame = self.frame[index]
            target = self.target[index]
            em_tar = self.em_tar[index]

        self.check_completeness(False)

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


if __name__ == '__main__':
    pass
