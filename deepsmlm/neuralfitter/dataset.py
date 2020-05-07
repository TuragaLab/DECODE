import time

import torch
from torch.utils.data import Dataset


class SMLMDataset(Dataset):
    _pad_modes = (None, 'same')

    def __init__(self, *, em_proc, frame_proc, bg_frame_proc, tar_gen, weight_gen,
                 frame_window: int, pad: str = None, return_em: bool):
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

        if self._frames.size(1) != 1:
            raise ValueError("Frames must be one-channeled, i.e. N x C=1 x H x W.")

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

        """Pad index, get frames and emitters."""
        index = self._pad_index(index)
        frames = self._get_frames(self._frames, index).squeeze(1)
        bg_frame = self._bg_frames[index] if self._bg_frames is not None else None
        tar_emitter = self._emitter[index]

        """Process Emitters"""
        frames, target, weight, tar_emitter = self._process_sample(frames, tar_emitter, bg_frame)

        return self._return_sample(frames, target.squeeze(0), weight.squeeze(0), tar_emitter)


class SMLMLiveDataset(SMLMDataset):

    def __init__(self, *, simulator, em_proc, frame_proc, bg_frame_proc, tar_gen, weight_gen, frame_window, pad,
                 return_em=False):

        super().__init__(em_proc=em_proc, frame_proc=frame_proc, bg_frame_proc=bg_frame_proc,
                         tar_gen=tar_gen, weight_gen=weight_gen,
                         frame_window=frame_window, pad=pad, return_em=return_em)

        self.simulator = simulator
        self._bg_frames = None

    def sanity_check(self):

        super().sanity_check()
        if self._emitter is not None and not isinstance(self._emitter, (list, tuple)):
            raise TypeError("EmitterSet shall be stored in list format, where each list item is one target emitter.")

    def sample(self, verbose=False):

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

    def __getitem__(self, ix):

        ix = self._pad_index(ix)

        tar_emitter = self._emitter[ix]
        frames = self._get_frames(self._frames, ix)
        bg_frame = self._bg_frames[[ix]]

        frames, target, weight, tar_emitter = self._process_sample(frames, tar_emitter, bg_frame)

        return self._return_sample(frames, target.squeeze(0), weight.squeeze(0), tar_emitter)


class InferenceDataset(SMLMStaticDataset):
    """
    A SMLM dataset without ground truth data. This is dummy wrapper to keep the visual appearance of a separate dataset.
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

    def __getitem__(self, index):
        """
        Get an inference sample.

        Args:
           index (int): index

        Returns:
           frames (torch.Tensor): processed frames. C x H x W
        """
        index = self._pad_index(index)

        frames = self._get_frames(self._frames, index).squeeze(1)

        if self.frame_proc:
            frames = self.frame_proc.forward(frames)

        return frames


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
