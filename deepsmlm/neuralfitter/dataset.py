import ctypes
import multiprocessing as mp
import numpy as np
import torch
import tqdm
from torch.utils.data import Dataset

from deepsmlm.generic.emitter import EmitterSet


class SMLMStaticDataset(Dataset):
    """
    A simple and static SMLMDataset.

    Attributes:
        fwindow (int): width of frame window

        tar_gen: target generator function
        frame_proc: frame processing function
        em_proc: emitter processing / filter function

        return_em (bool): return EmitterSet in getitem method.
    """

    def __init__(self, *, frames, em, tar_gen, frame_proc, em_proc, fwindow=3, return_em=True):
        """

        Args:
            frames (torch.Tensor): frames. N x C x H x W
            em (list of EmitterSets): ground-truth emitter-sets
            tar_gen: target generator function
            frame_proc: frame processing function
            em_proc: emitter processing / filter function
            fwindow (int): width of frame window
            return_em (bool): return EmitterSet in getitem method.
        """

        super().__init__()

        self._frames = frames
        self._em = em

        self.frame_proc = frame_proc
        self.em_proc = em_proc
        self.tar_gen = tar_gen
        self.multi_frame = fwindow if fwindow is not None else 1  # otherwise the slicing logic does not work

        self.return_em = return_em

        """Sanity checks."""
        if self.multi_frame is not None and self.multi_frame % 2 != 1:
            raise ValueError("Multi-Frame must be None or odd integer.")

        if self._em is not None and not isinstance(self._em, (list, tuple)):
            raise ValueError("EM must be None, list or tuple.")

        if self._frames.size(1) != 1:
            raise ValueError("Frames must be one-channeled, i.e. N x C=1 x H x W.")

    def __len__(self):
        return self._frames.size(0)

    def _get_frame_window(self, index):
        """
        Get frames in a given window and pad with same.

        Args:
            index:

        Returns:
            frames:

        """
        hw = (self.multi_frame - 1) // 2  # half window without centre
        frame_ix = torch.arange(index - hw, index + hw + 1).clamp(0, len(self) - 1)
        frames = self._frames[frame_ix].squeeze(1)
        return frames

    def __getitem__(self, index):
        """
        Get a training sample.

        Args:
            index (int): index

        Returns:
            frames (torch.Tensor): processed frames. C x H x W
            tar (torch.Tensor): target
            em_tar (optional): Ground truth emitters

        """
        """Get frames. Pad with same."""
        frames = self._get_frame_window(index)

        if self.frame_proc:
            frames = self.frame_proc.forward(frames)

        """Process Emitters"""
        em_tar = self._em[index] if self._em is not None else None
        if self.em_proc:
            em_tar = self.em_proc.forward(em_tar)

        """Generate target"""
        if self.tar_gen:
            tar = self.tar_gen.forward(em_tar)

        if self.return_em:
            return frames, tar, em_tar
        else:
            if self.tar_gen:
                return frames, tar
            else:
                return frames


class InferenceDataset(SMLMStaticDataset):
    """
    A SMLM dataset without ground truth data. This is dummy wrapper to keep the visual appearance of a separate dataset.
    """
    def __init__(self, *, frames, frame_proc, fwindow):
        """

        Args:
            frames (torch.Tensor): frames
            frame_proc: frame processing function
            fwindow (int): frame window
        """
        super().__init__(frames=frames, em=None, tar_gen=None, frame_proc=frame_proc, em_proc=None,
                         fwindow=fwindow, return_em=False)

    def __getitem__(self, index):
        """
        Get an inference sample.

        Args:
           index (int): index

        Returns:
           frames (torch.Tensor): processed frames. C x H x W
        """
        frames = self._get_frame_window(index)

        if self.frame_proc:
            frames = self.frame_proc.forward(frames)

        return frames


class SMLMTrainingEngineDataset(Dataset):
    def __init__(self, engine,
                 em_filter, input_prep, target_gen, weight_gen, return_em_tar=False):
        """

        Args:
            engine: (SMLMTrainingEngine)
            em_filter: (callable) that filters the emitters as provided by the simulation engine
            input_prep: (callable) that prepares the input data for the network (e.g. rescaling)
            target_gen: (callable) that generates the training data
            weight_gen: (callable) that generates a weight mask corresponding to the target / output data
            return_em_tar: (bool) return target emitters in the form of an emitter set. use for test set

        """
        self._engine = engine
        self._x_in = None  # camera frames
        self._tar_em = None  # emitter target
        self._aux = None  # auxiliary things

        self.em_filter = em_filter
        self.input_prep = input_prep
        self.target_gen = target_gen
        self.weight_gen = weight_gen
        self.return_em_tar = return_em_tar

    def load_from_engine(self):
        """
        Gets the data from the engine makes basic transformations

        Returns:
            None

        """

        self._tar_em = None
        self._x_in = None
        self._aux = None

        data = self._engine.load_and_touch()
        self._tar_em = data[0]
        self._x_in = data[1]
        if len(data) >= 3:
            self._aux = data[2:]
        else:
            self._aux = [None] * self._x_in.size(0)

    def __len__(self):
        return self._x_in.size(0)

    def __getitem__(self, ix):
        """

        Args:
            ix: (int) item index

        Returns:
            x_in: (torch.Tensor) input frame
            tar_frame: (torch.Tensor) target frame
            tar_em: (EmitterSet, optional) target emitter

        """

        """Get a single sample from the list."""
        x_in = self._x_in[ix]
        tar_em = self._tar_em[ix]
        aux = [a[ix] for a in self._aux]

        """Preparation on input, emitter filtering, target generation"""
        x_in = self.input_prep.forward_(x_in)
        tar_em = self.em_filter.forward_(tar_em)
        tar_frame = self.target_gen.forward_(tar_em, *aux)
        weight = self.weight_gen.forward_(x_in, tar_em, *aux)

        if not self.return_em_tar:
            return x_in, tar_frame, weight
        else:
            return x_in, tar_frame, weight, tar_em


class SMLMSimulationDatasetOnFly(Dataset):
    """
    Simple implementation of a dataset which can generate samples from a simulator and returns them along with the
    emitters that are on the frame.
    I did this mainly here because I did not want to care about the multiprocessing myself and rather use the pytorch
    dataset thingy which does that for me.
    In itself this class will not be used for training a network directly.
    """

    def __init__(self, simulator, ds_size: int):
        """

        Args:
            simulator: (Simulation) (in principle anything with a) forward method
            ds_size: (int) size of the dataset
        """
        super().__init__()
        self.sim = simulator
        self.ds_size = ds_size

        # make sure that simulator has a prior to sample from and not a static emitter set
        assert not isinstance(self.sim.em, EmitterSet)

    def __len__(self):
        return self.ds_size

    def __getitem__(self, item):
        """
        Returns the items

        Args:
            item: (int) index of sample

        Returns:
            em_tar: emitter target
            cam_frames: camera frames
            bg_frames: background frames
        """

        cam_frames, bg_frames, em_tar = self.sim.forward_()
        return em_tar, cam_frames, bg_frames


class SMLMDatasetOnFly(Dataset):
    def __init__(self, extent, prior, simulator, ds_size, in_prep, tar_gen, w_gen, return_em_tar=False,
                 predict_bg=True):
        """

        :param extent:
        :param prior:
        :param simulator:
        :param ds_size:
        :param in_prep: Prepare input to NN. Any instance with forwrard method
        :param tar_gen: Generate target for learning.
        :param static:
        :param lifetime:
        :param return_em_tar: __getitem__ method returns em_target
        """
        super().__init__()

        self.extent = extent
        self.data_set_size = ds_size

        self.return_em_tar = return_em_tar
        self.predict_bg = predict_bg
        self.prior = prior
        self.simulator = simulator

        self.input_preperator = in_prep  # N2C()
        self.target_generator = tar_gen
        self.weight_generator = w_gen

    def pop_new(self):
        """
        :return: emitter (all three frames)
                 frames
                 target frames
                 emitters on the target frame (i.e. the middle frame)
        """
        emitter = self.prior.pop()
        frame_sim, bg_sim = self.simulator.forward_(emitter)
        frame = self.input_preperator.forward_(frame_sim)  # C x H x W
        emitter_on_tar_frame = emitter.get_subset_frame(0, 0)

        if self.weight_generator is not None:
            # generate the weight mask
            weight_mask = self.weight_generator.forward_(frame, emitter_on_tar_frame, bg_sim[0])
        else:
            weight_mask = torch.zeros(0)

        if self.predict_bg and self.target_generator is not None:
            target = self.target_generator.forward_(emitter_on_tar_frame, bg_sim)
        elif self.target_generator is not None:
            target = self.target_generator.forward_(emitter_on_tar_frame)
        else:
            target = torch.zeros(0)

        return emitter, frame, target, weight_mask, emitter_on_tar_frame

    def __len__(self):
        return self.data_set_size

    def __getitem__(self, index):

        emitter, frame, target, weight_mask, em_tar = self.pop_new()

        """Make sure the data types are correct"""
        self._check_datatypes(frame, target)

        if self.return_em_tar:
            return frame, target, weight_mask, em_tar
        return frame, target, weight_mask

    @staticmethod
    def _check_datatypes(*args, none_okay=True):

        for arg in args:
            if isinstance(arg, torch.FloatTensor):
                continue
            if (arg is None or arg == [None]) and none_okay:
                continue
            raise ValueError(f"At least one of the tensors in the dataset is of wrong type. The datatype is "
                             f"{type(arg)}")

    def step(self):
        pass


class SMLMDatasetOneTimer(SMLMDatasetOnFly):
    def __init__(self, extent, prior, simulator, ds_size, in_prep, tar_gen, w_gen, return_em_tar=False,
                 predict_bg=True):
        super().__init__(extent, prior, simulator, ds_size, in_prep, tar_gen, w_gen, return_em_tar, predict_bg)

        self.frame = [None] * self.__len__()
        self.target = [None] * self.__len__()
        self.weight_mask = [None] * self.__len__()
        self.em_tar = [None] * self.__len__()

        """Pre-Calculcate the complete dataset and use the same data as one draws samples.
                This is useful for the testset or the classical deep learning feeling of not limited training data."""
        for i in tqdm.trange(self.__len__(), desc='Pre-Calculate Dataset'):
            _, frame, target, weight_mask, em_tar = self.pop_new()
            self.frame[i] = frame
            self.target[i] = target
            self.weight_mask[i] = weight_mask
            self.em_tar[i] = em_tar

    def get_gt_emitter(self, output_format='list'):
        """
        Get the complete ground truth. Should only be used for static data.
        :param output_format: either list (list of emittersets) or concatenated Emittersets.
        :return:
        """

        if output_format == 'list':
            return self.em_tar
        elif output_format == 'cat':
            return EmitterSet.cat(self.em_tar, step_frame_ix=1)

    def __getitem__(self, index):

        self._check_datatypes(self.frame[index], self.target[index])

        if self.return_em_tar:
            return self.frame[index], self.target[index], self.weight_mask[index], self.em_tar[index]
        return self.frame[index], self.target[index], self.weight_mask[index]


class SMLMDatasetOnFlyCached(SMLMDatasetOnFly):
    def __init__(self, extent, prior, simulator, ds_size, in_prep, tar_gen, w_gen, lifetime, return_em_tar=False,
                 predict_bg=True):

        super().__init__(extent, prior, simulator, ds_size, in_prep, tar_gen, w_gen, return_em_tar, predict_bg)

        self.lifetime = lifetime
        self.time_til_death = lifetime

        """Initialise Frame and Target. Call drop method to create list."""
        _, frame_dummy, target_dummy, weight_dummy, _ = self.pop_new()

        frames_base = mp.Array(ctypes.c_float, self.__len__() * frame_dummy.numel())
        frames = np.ctypeslib.as_array(frames_base.get_obj())
        frames = frames.reshape(self.__len__(), frame_dummy.size(0), frame_dummy.size(1), frame_dummy.size(2))

        target_base = mp.Array(ctypes.c_float, self.__len__() * target_dummy.numel())
        target = np.ctypeslib.as_array(target_base.get_obj())
        target = target.reshape(self.__len__(), target_dummy.size(0), target_dummy.size(1), target_dummy.size(2))

        weight_base = mp.Array(ctypes.c_float, self.__len__() * weight_dummy.numel())
        weight = np.ctypeslib.as_array(weight_base.get_obj())
        weight = weight.reshape(self.__len__(), weight_dummy.size(0), weight_dummy.size(1), weight_dummy.size(2))

        self.frame = torch.from_numpy(frames)
        self.target = torch.from_numpy(target)
        self.weight_mask = torch.from_numpy(weight)
        self.em_tar = [None] * self.__len__()
        self.use_cache = False

        self.drop_data_set(verbose=False)

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
        self.weight_mask *= float('nan')
        self.em_tar = [None] * self.__len__()

        self.use_cache = False
        self.time_til_death = self.lifetime

        if verbose:
            print("Dataset dropped. Will calculate a new one in next epoch.")

    def __getitem__(self, index):

        if not self.use_cache:
            emitter, frame, target, weight_mask, em_tar = self.pop_new()
            self.frame[index] = frame
            self.target[index] = target
            self.weight_mask[index] = weight_mask
            self.em_tar[index] = em_tar

        frame, target, weight_mask, em_tar = self.frame[index], self.target[index], self.weight_mask[index], \
                                             self.em_tar[index]

        """Make sure the data types are correct"""
        self._check_datatypes(frame, target)

        if self.return_em_tar:
            return frame, target, weight_mask, em_tar
        else:
            return frame, target, weight_mask



