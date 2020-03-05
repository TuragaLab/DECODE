import ctypes
import multiprocessing as mp

import numpy as np
import torch
import tqdm
from torch.utils.data import Dataset


class SMLMStaticDataset(Dataset):
    """
    A simple and static SMLMDataset.

    Attributes:
        f_win (int): width of frame window

        tar_gen: target generator function
        frame_proc: frame processing function
        em_proc: emitter processing / filter function
        weight_gen: weight generator function

        return_em (bool): return EmitterSet in getitem method.
    """

    def __init__(self, *, frames, em, frame_proc, em_proc, tar_gen, weight_gen=None, f_win=3, return_em=True):
        """

        Args:
            frames (torch.Tensor): frames. N x C x H x W
            em (list of EmitterSets): ground-truth emitter-sets
            frame_proc: frame processing function
            em_proc: emitter processing / filter function
            tar_gen: target generator function
            weight_gen: weight generator function
            f_win (int): width of frame window
            return_em (bool): return EmitterSet in getitem method.
        """

        super().__init__()

        self._frames = frames
        self._em = em

        self.frame_proc = frame_proc
        self.em_proc = em_proc
        self.tar_gen = tar_gen
        self.weight_gen = weight_gen
        self.multi_frame = f_win if f_win is not None else 1  # otherwise the slicing logic does not work

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
        tar = self.tar_gen.forward(em_tar)

        """Generate weight mask"""
        if self.weight_gen:
            weight = self.weight_gen.forward(frames, em_tar)
        else:
            weight = None

        if self.return_em:
            return frames, tar, weight, em_tar

        return frames, tar, weight


class InferenceDataset(SMLMStaticDataset):
    """
    A SMLM dataset without ground truth data. This is dummy wrapper to keep the visual appearance of a separate dataset.
    """

    def __init__(self, *, frames, frame_proc, f_win=3):
        """

        Args:
            frames (torch.Tensor): frames
            frame_proc: frame processing function
            f_win (int): frame window
        """
        super().__init__(frames=frames, em=None, frame_proc=frame_proc, em_proc=None, tar_gen=None, f_win=f_win,
                         return_em=False)

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
    """
    A dataset to use in conjunction with the training engine. It serves the purpose to load the data from the engine.

    Attributes:

    tar_gen: target generator function
    frame_proc: frame processing function
    em_proc: emitter processing / filter function
    weight_gen: weight generator function

    return_em (bool): return EmitterSet in getitem method.

    """
    def __init__(self, *, engine, em_proc, frame_proc, tar_gen, weight_gen, return_em=False):
        """

        Args:
            engine: (SMLMTrainingEngine)
            em_proc: (callable) that filters the emitters as provided by the simulation engine
            frame_proc: (callable) that prepares the input data for the network (e.g. rescaling)
            tar_gen: (callable) that generates the training data
            weight_gen: (callable) that generates a weight mask corresponding to the target / output data
            return_em: (bool) return target emitters in the form of an emitter set. use for test set

        """
        self._engine = engine
        self._x_in = None  # camera frames
        self._tar_em = None  # emitter target
        self._aux = None  # auxiliary things

        self.frame_proc = frame_proc
        self.em_proc = em_proc
        self.tar_gen = tar_gen
        self.weight_gen = weight_gen

        self.return_em = return_em

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
        if len(data) >= 3:  # auxiliary stuff is appended at the end
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
        x_in = self.frame_proc.forward(x_in)
        tar_em = self.em_proc.forward(tar_em)
        tar_frame = self.tar_gen.forward(tar_em, *aux)
        weight = self.weight_gen.forward(x_in, tar_em, *aux)

        if not self.return_em:
            return x_in, tar_frame, weight
        else:
            return x_in, tar_frame, weight, tar_em


class SMLMDatasetOnFly(Dataset):
    """
    A dataset in which the samples are generated on the fly.

    Attributes:
        ds_size (int): dataset (pseudo) size
        prior: something with a pop() method to pop new samples
        simulator:

    """
    def __init__(self, *, prior, simulator, ds_size: int, frame_proc, em_proc, tar_gen, weight_gen,
                 return_em: bool =False):
        super().__init__()

        self.ds_size = ds_size

        self.prior = prior
        self.simulator = simulator

        self.frame_proc = frame_proc
        self.em_proc = em_proc
        self.tar_gen = tar_gen
        self.weight_gen = weight_gen

        self.return_em = return_em

    def _pop_sample(self):
        """
        Generate new training sample

        Returns:
            emitter: all emitters
            tar_em: target emitters (emitters on the central frame)
            frame: input frames
            target: target frames
            weight: weight "frame"
        """

        emitter = self.prior.pop()  # pop new emitters (on all frames, those are not necessarily the target emitters)

        frame, bg, _ = self.simulator.forward(emitter)
        frame = self.frame_proc.forward(frame)
        tar_em = emitter.get_subset_frame(0, 0)  # target emitters

        if self.weight_gen is not None:
            weight = self.weight_gen.forward(frame, tar_em, bg)
        else:
            weight = None

        target = self.tar_gen.forward_(tar_em)

        return emitter, tar_em, frame, target, weight

    def __len__(self):
        return self.ds_size

    def __getitem__(self, index):

        all_em, tar_em, frame, target, weight = self._pop_sample()

        if self.return_em:
            return frame, target, weight, tar_em

        return frame, target, weight


class SMLMDatasetOneTimer(SMLMDatasetOnFly):
    def __init__(self, *, prior, simulator, ds_size, frame_proc, em_proc, tar_gen, weight_gen, return_em=False):
        super().__init__(prior=prior, simulator=simulator, ds_size=ds_size, frame_proc=frame_proc, em_proc=em_proc,
                         tar_gen=tar_gen, weight_gen=weight_gen, return_em=return_em)

        self.frame = [None] * len(self)
        self.target = [None] * len(self)
        self.weight_mask = [None] * len(self)
        self.tar_em = [None] * len(self)

        """
        Pre-calculate the complete dataset and use the same data as one draws samples.
        This is useful for the testset or the classical deep learning feeling of limited training data.
        """
        for i in tqdm.trange(self.__len__(), desc='Pre-Calculate Dataset'):
            _, tar_em, frame, target, weight_mask = self._pop_sample()
            self.tar_em[i] = tar_em
            self.frame[i] = frame
            self.target[i] = target
            self.weight_mask[i] = weight_mask

    def __getitem__(self, index):

        if self.return_em:
            return self.frame[index], self.target[index], self.weight_mask[index], self.tar_em[index]

        return self.frame[index], self.target[index], self.weight_mask[index]


class SMLMDatasetOnFlyCached(SMLMDatasetOnFly):
    def __init__(self, *, prior, simulator, ds_size, lifetime, frame_proc, em_proc, tar_gen, weight_gen,
                 return_em=False):

        super().__init__(prior=prior, simulator=simulator, ds_size=ds_size, frame_proc=frame_proc, em_proc=em_proc,
                         tar_gen=tar_gen, weight_gen=weight_gen, return_em=return_em)

        self.lifetime = lifetime
        self.time_til_death = lifetime

        """Initialise Frame and Target Cache. Call drop method to create list."""
        _, frame_dummy, target_dummy, weight_dummy, _ = self._pop_sample()

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

    def drop_data_set(self, verbose=True):
        """
        Invalidate / clear cache.
        Args:
            verbose (bool): print to console when dropped
        """

        self.frame *= float('nan')
        self.target *= float('nan')
        self.weight_mask *= float('nan')
        self.em_tar = [None] * len(self)

        self.use_cache = False
        self.time_til_death = self.lifetime

        if verbose:
            print("Dataset dropped. Will calculate a new one in next epoch.")

    def step(self):
        """
        Reduces lifetime by one step

        """
        self.time_til_death -= 1
        if self.time_til_death <= 0:
            self.drop_data_set()
        else:
            self.use_cache = True

    def __getitem__(self, index):

        if not self.use_cache:
            emitter, frame, target, weight_mask, em_tar = self._pop_sample()
            self.frame[index] = frame
            self.target[index] = target
            self.weight_mask[index] = weight_mask
            self.em_tar[index] = em_tar

        frame, target, weight_mask, em_tar = self.frame[index], self.target[index], self.weight_mask[index], \
                                             self.em_tar[index]

        if self.return_em:
            return frame, target, weight_mask, em_tar

        return frame, target, weight_mask
