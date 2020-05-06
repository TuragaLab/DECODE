from abc import ABC, abstractmethod


class ProcessEmitters(ABC):
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


class Identity(ProcessEmitters):
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


class RemoveOutOfFOV(ProcessEmitters):
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

    def clean_emitter(self, xyz):
        """
        Returns index of emitters that are inside the specified extent.

        Args:
            xyz:

        Returns:

        """

        is_emit = (xyz[:, 0] >= self.xextent[0]) * (xyz[:, 0] < self.xextent[1]) * \
                  (xyz[:, 1] >= self.yextent[0]) * (xyz[:, 1] < self.yextent[1])

        if self.zextent is not None:
            is_emit *= (xyz[:, 2] >= self.zextent[0]) * (xyz[:, 2] < self.zextent[1])

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
        is_emit = self.clean_emitter(em_mat)

        return em_set[is_emit]
