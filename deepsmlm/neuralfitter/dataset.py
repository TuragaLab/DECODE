import time
from deprecated import deprecated

import torch
from torch.utils.data import Dataset


class SMLMDataset(Dataset):
    """
    SMLM base dataset.


    """
    _pad_modes = (None, 'same')

    def __init__(self, *, em_proc, frame_proc, bg_frame_proc, tar_gen, weight_gen,
                 frame_window: int, pad: str = None, return_em: bool):
        """
        Init new dataset.

        Args:
            em_proc: Emitter processing
            frame_proc: Frame processing
            bg_frame_proc: Background frame processing
            tar_gen: Target generator
            weight_gen: Weight generator
            frame_window: number of frames per sample / size of frame window
            pad: pad mode, applicable for first few, last few frames (relevant when frame window is used)
            return_em: return target emitter

        """
        super().__init__()

        self._frames = None
        self._emitter = None

        self.em_proc = em_proc
        self.frame_proc = frame_proc
        self.bg_frame_proc = bg_frame_proc
        self.tar_gen = tar_gen
        self.weight_gen = weight_gen

        self.frame_window = frame_window
        self.pad = pad
        self.return_em = return_em

        """Sanity"""
        self.sanity_check()

    def __len__(self):
        if self.pad is None:  # loosing samples at the border
            return self._frames.size(0) - self.frame_window + 1

        elif self.pad == 'same':
            return self._frames.size(0)

    def sanity_check(self):
        """
        Checks the sanity of the dataset, if fails, errors are raised.

        """

        if self.pad not in self._pad_modes:
            raise ValueError(f"Pad mode {self.pad} not available. Available pad modes are {self._pad_modes}.")

        if self.frame_window is not None and self.frame_window % 2 != 1:
            raise ValueError(f"Unsupported frame window. Frame window must be odd integered, not {self.frame_window}.")

    def _get_frames(self, frames, index):
        hw = (self.frame_window - 1) // 2  # half window without centre

        frame_ix = torch.arange(index - hw, index + hw + 1).clamp(0, len(frames) - 1)
        return frames[frame_ix]

    def _pad_index(self, index):

        if self.pad is None:
            assert index >= 0, "Negative indexing not supported."
            return index + (self.frame_window - 1) // 2

        elif self.pad == 'same':
            return index

    def _process_sample(self, frames, tar_emitter, bg_frame):

        """Process"""
        if self.frame_proc is not None:
            frames = self.frame_proc.forward(frames)

        if self.bg_frame_proc is not None:
            bg_frame = self.bg_frame_proc.forward(bg_frame)

        if self.em_proc is not None:
            tar_emitter = self.em_proc.forward(tar_emitter)

        if self.tar_gen is not None:
            target = self.tar_gen.forward(tar_emitter, bg_frame)
        else:
            target = None

        if self.weight_gen is not None:
            weight = self.weight_gen.forward(target, tar_emitter, bg_frame)
        else:
            weight = None

        return frames, target, weight, tar_emitter

    def _return_sample(self, frame, target, weight, emitter):

        if self.return_em:
            return frame, target, weight, emitter
        else:
            return frame, target, weight


class SMLMStaticDataset(SMLMDataset):
    """
    A simple and static SMLMDataset.

    Attributes:
        frame_window (int): width of frame window

        tar_gen: target generator function
        frame_proc: frame processing function
        em_proc: emitter processing / filter function
        weight_gen: weight generator function

        return_em (bool): return EmitterSet in getitem method.
    """

    def __init__(self, *, frames, emitter: (None, list, tuple), frame_proc, bg_frame_proc, em_proc, tar_gen,
                 bg_frames=None, weight_gen=None, frame_window=3, pad: (str, None) = None, return_em=True):
        """

        Args:
            frames (torch.Tensor): frames. N x C x H x W
            em (list of EmitterSets): ground-truth emitter-sets
            frame_proc: frame processing function
            em_proc: emitter processing / filter function
            tar_gen: target generator function
            weight_gen: weight generator function
            frame_window (int): width of frame window
            return_em (bool): return EmitterSet in getitem method.
        """

        super().__init__(em_proc=em_proc, frame_proc=frame_proc, bg_frame_proc=bg_frame_proc,
                         tar_gen=tar_gen, weight_gen=weight_gen,
                         frame_window=frame_window, pad=pad, return_em=return_em)

        self._frames = frames
        self._emitter = emitter
        self._bg_frames = bg_frames

        if self._frames is not None and self._frames.dim() != 3:
            raise ValueError("Frames must be 3 dimensional, i.e. N x H x W.")

    def __getitem__(self, ix):
        """
        Get a training sample.

        Args:
            ix (int): index

        Returns:
            frames (torch.Tensor): processed frames. C x H x W
            tar (torch.Tensor): target
            em_tar (optional): Ground truth emitters

        """

        """Pad index, get frames and emitters."""
        ix = self._pad_index(ix)

        tar_emitter = self._emitter[ix] if self._emitter is not None else None
        frames = self._get_frames(self._frames, ix)
        bg_frame = self._bg_frames[ix] if self._bg_frames is not None else None

        frames, target, weight, tar_emitter = self._process_sample(frames, tar_emitter, bg_frame)

        return self._return_sample(frames, target, weight, tar_emitter)


class InferenceDataset(SMLMStaticDataset):
    """
    A SMLM dataset without ground truth data.
    This is dummy wrapper to keep the visual appearance of a separate dataset.
    """

    def __init__(self, *, frames, frame_proc, frame_window):
        """

        Args:
            frames (torch.Tensor): frames
            frame_proc: frame processing function
            frame_window (int): frame window
        """
        super().__init__(frames=frames, emitter=None, frame_proc=frame_proc, bg_frame_proc=None, em_proc=None,
                         tar_gen=None, pad='same', frame_window=frame_window, return_em=False)

    def _return_sample(self, frame, target, weight, emitter):
        return frame


class SMLMLiveDataset(SMLMStaticDataset):
    """
    A SMLM dataset where new datasets is sampleable via the sample() method.

    """

    def __init__(self, *, simulator, em_proc, frame_proc, bg_frame_proc, tar_gen, weight_gen, frame_window, pad,
                 return_em=False):

        super().__init__(emitter=None, frames=None,
                         em_proc=em_proc, frame_proc=frame_proc, bg_frame_proc=bg_frame_proc,
                         tar_gen=tar_gen, weight_gen=weight_gen,
                         frame_window=frame_window, pad=pad, return_em=return_em)

        self.simulator = simulator
        self._bg_frames = None

    def sanity_check(self):

        super().sanity_check()
        if self._emitter is not None and not isinstance(self._emitter, (list, tuple)):
            raise TypeError("EmitterSet shall be stored in list format, where each list item is one target emitter.")

    def sample(self, verbose: bool = False):
        """
        Sample new acquisition, i.e. a whole dataset.

        Args:
            verbose: print performance / verification information

        """

        def set_frame_ix(em):  # helper function
            em.frame_ix = torch.zeros_like(em.frame_ix)
            return em

        """Sample new dataset."""
        t0 = time.time()
        emitter, frames, bg_frames = self.simulator.sample()
        if verbose:
            print(f"Sampled dataset in {time.time() - t0:.2f}s. {len(emitter)} emitters on {frames.size(0)} frames.")

        """Split Emitters into list of emitters (per frame) and set frame_ix to 0."""
        emitter = emitter.split_in_frames(0, frames.size(0) - 1)
        emitter = [set_frame_ix(em) for em in emitter]

        self._emitter = emitter
        self._frames = frames.cpu()
        self._bg_frames = bg_frames.cpu()


class SMLMLiveSampleDataset(SMLMDataset):
    """
    A SMLM dataset where a new sample is drawn per (training) sample.

    """

    def __init__(self, *, simulator, ds_len, em_proc, frame_proc, bg_frame_proc, tar_gen, weight_gen, frame_window, return_em=False):
        super().__init__(em_proc=em_proc, frame_proc=frame_proc, bg_frame_proc=bg_frame_proc,
                         tar_gen=tar_gen, weight_gen=weight_gen,
                         frame_window=frame_window, pad=None, return_em=return_em)

        self.simulator = simulator
        self.ds_len = ds_len

    def __len__(self):
        return self.ds_len

    def __getitem__(self, ix):

        """Sample"""
        emitter, frames, bg_frames = self.simulator.sample()

        assert frames.size(0) % 2 == 1
        frames = self._get_frames(frames, (frames.size(0) - 1) // 2)
        tar_emitter = emitter.get_subset_frame(0, 0)  # target emitters are the zero ones
        bg_frames = bg_frames[1]

        frames, target, weight, tar_emitter = self._process_sample(frames, tar_emitter, bg_frames)

        return self._return_sample(frames, target, weight, tar_emitter)


@deprecated(reason="Not needed anymore.")
class SMLMSampleStreamEngineDataset(SMLMDataset):
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
        super().__init__(em_proc=em_proc, frame_proc=frame_proc, bg_frame_proc=None,
                         tar_gen=tar_gen, weight_gen=weight_gen, frame_window=None,
                         pad=None, return_em=return_em)

        self._engine = engine
        self._x_in = None  # camera frames
        self._tar_em = None  # emitter target
        self._aux = None  # auxiliary things

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
        weight = self.weight_gen.forward(tar_frame, tar_em, *aux)

        return self._return_sample(x_in, tar_frame, weight, tar_em)


@deprecated(reason="Not needed anymore.")
class SMLMDatasetEngineDataset(SMLMSampleStreamEngineDataset):

    def __init__(self, *, engine, em_proc, frame_proc, tar_gen, weight_gen, frame_window, pad=None, return_em=False):
        super().__init__(engine=engine, em_proc=em_proc, frame_proc=frame_proc, tar_gen=tar_gen, weight_gen=weight_gen,
                         return_em=return_em)

        self.frame_window = frame_window
        self.pad = pad

    def __len__(self):
        if self.pad is None:  # loosing samples at the border
            return self._x_in.size(0) - self.frame_window + 1
        elif self.pad == 'same':
            return self._x_in.size(0)

    def load_from_engine(self):
        def set_frame_ix(em):
            em.frame_ix = torch.zeros_like(em.frame_ix)
            return em

        super().load_from_engine()

        """
        It's more efficient to write an entire frame set to binary instead of a list of emittersets (per frame).
        However, getting a subset of the frameset each time is expensive, since search starts over and over.
        Therefore, split the EmitterSet after loading and move the indices to 0 (since the target emitters per
        example are expected to be 0).
        """
        if not isinstance(self._tar_em, (list, tuple)):
            self._tar_em = self._tar_em.split_in_frames(0, self._x_in.size(0) - 1)
            self._tar_em = [set_frame_ix(em) for em in self._tar_em]

    def __getitem__(self, index):

        index = self._pad_index(index)

        x_in = self._get_frames(self._x_in, index)
        tar_em = self._tar_em[index]
        aux = [a[index] for a in self._aux]

        """Give Frames and BG a channel dimension"""
        assert len(aux) == 1, "Auxiliary input can only be background."
        aux[0].unsqueeze_(0)

        """Preparation on input, emitter filtering, target generation"""
        x_in = self.frame_proc.forward(x_in)
        tar_em = self.em_proc.forward(tar_em)
        tar_frame = self.tar_gen.forward(tar_em, *aux) if self.tar_gen is not None else None
        weight = self.weight_gen.forward(tar_frame, tar_em, *aux) if self.weight_gen is not None else None

        return self._return_sample(x_in, tar_frame, weight, tar_em)
