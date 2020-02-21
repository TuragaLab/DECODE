from abc import ABC, abstractmethod
import torch
import torch.nn
import deprecated as depr
import warnings

from ..generic.emitter import EmitterSet


class PreProcessing(ABC):
    def __init__(self):
        super().__init__()

    def __call__(self, *args, **kwargs):
        """
        Convenience around forward.

        Args:
            *args:
            **kwargs:

        Returns:

        """
        self.forward(*args, **kwargs)

    @abstractmethod
    def forward(self, *args, **kwargs):
        """
        All derived classes must implement a forward method that does not change the input inplace and implements
        some kind of processing. In most cases the return type should be the same type as the (first) input argument.

        Args:
            *args:
            **kwargs:

        Returns:

        """
        return


class Identity(PreProcessing):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        """
        The do nothing pre-processing.

        Args:
            x: arbitrary

        Returns:
            x:
        """
        return x


class RemoveOutOfFOV(PreProcessing):
    def __init__(self, xextent, yextent, zextent=None):
        """
        Processing class to remove emitters that are outside a specified extent.
        The lower / left respective extent limits are included, the right / upper extent limit is excluded / open.

        Args:
            xextent: extent of allowed field in x direction
            yextent: extent of allowed field in y direction
            zextent: (optional) extent of allowed field in z direction
        """
        super().__init__()

        self.xextent = xextent
        self.yextent = yextent
        self.zextent = zextent

    def _clean_emitter(self, xyz):
        """
        Returns index of emitters that are inside the specified extent.

        Args:
            xyz:

        Returns:

        """

        is_emit = (xyz[:, 0] >= self.xextent[0]) * (xyz[:, 0] < self.xextent[1]) * \
                  (xyz[:, 1] >= self.yextent[0]) * (xyz[:, 1] < self.yextent[1])

        if self.zextent is not None:
            is_emit *= (xyz[:, 2] >= self.zextent[0]) * (xyz[:, 2] >= self.zextent[1])

        return is_emit

    def forward(self, em_set):
        """
        Removes emitters that are outside of the specified extent.

        Args:
            em_set:

        Returns:
            EmitterSet
        """
        em_mat = em_set.xyz
        is_emit = self._clean_emitter(em_mat)

        return em_set[is_emit]


"""------------------------------------         Deprecation Candidates  ----------------------------------"""


class N2C(PreProcessing):
    """
    Change from Batch to channel dimension.
    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        warnings.warn("Deprecation candidate.", DeprecationWarning)
        if isinstance(x, tuple) or isinstance(x, list):
            out = [None] * x.__len__()
            for i in range(x.__len__()):
                out[i] = self.forward(x[i])
            return out

        in_tensor = super().forward(x)
        if in_tensor.shape[1] != 1:
            raise ValueError("Shape is wrong.")
        return in_tensor.squeeze(1)


"""------------------------------------         Deprecated old stuff.   ------------------------------------"""


@depr.deprecated(version='0.1', reason="Refactoring.")
class DiscardBackground(PreProcessing):
    """
    A simple class which discards the background which comes out of the simulator because this will be target not input.
    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        """
        If x is tuple, second element (bg) will be discarded. If not nothing happens.
        :param x: tuple or tensor.
        :return: tensor
        """
        if not isinstance(x, torch.Tensor):
            return x[0]
        else:
            return x


@depr.deprecated(version='0.1', reason="Refactoring.")
class ThresholdPhotons:
    def __init__(self, photon_threshold, mode=None):
        """
        Thresholds the photon for prediction. Useful for CRLB calculation.

        Args:
            photon_threshold: threshold values for photon count
            mode: this makes it possible to use this as a pre-step for the weight generator and the target generator
        """
        self.photon_threshold = photon_threshold
        self._mode = mode

        if self._mode not in (None, 'target', 'weight'):
            raise ValueError("Not supported.")

    @staticmethod
    def parse(param, mode=None):
        return ThresholdPhotons(photon_threshold=param.HyperParameter.photon_threshold, mode=mode)

    def forward_impl(self, em):
        if self.photon_threshold is None:
            return em

        ix = em.phot >= self.photon_threshold
        return em[ix]

    def forward(self, *args):
        """
        Removes the emitters that have too few localisations.
        Cumbersome implementation because this can be used in multiple places.

        Args:
            args: various arguments. In standard / default mode just the emitterset.

        Returns:
            emitterset + args without the emitters with too low photon value

        """
        if self._mode is None:
            return self.forward_impl(args[0])

        elif self._mode == 'target':
            if args.__len__() == 1:
                return self.forward_impl(args[0])
            else:
                return (self.forward_impl(args[0]), *args[1:])

        elif self._mode == 'weight':
            if args.__len__() == 2:
                return args[0], self.forward_impl(args[1])
            else:
                return (args[0], self.forward_impl(args[1]), *args[2:])
