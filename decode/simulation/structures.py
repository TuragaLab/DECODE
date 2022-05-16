import torch

from abc import ABC, abstractmethod
from typing import Tuple


class StructurePrior(ABC):
    """
    Abstract structure which can be sampled from. All implementation / childs must define a 'pop' method and an area
    property that describes the area the structure occupies.

    """

    @property
    @abstractmethod
    def area(self) -> float:
        """
        Calculate the area which is occupied by the structure. This is useful to later calculate the density,
        and the effective number of emitters). This is the 2D projection. Not the volume.

        """
        raise NotImplementedError

    @abstractmethod
    def sample(self, n: int) -> torch.Tensor:
        """
        Sample n samples from structure.

        Args:
            n: number of samples

        """
        raise NotImplementedError


class RandomStructure(StructurePrior):
    """
    Random uniform 3D / 2D structure. As the name suggests, sampling from this structure gives samples from a 3D / 2D
    volume that origin from a uniform distribution.

    """

    def __init__(self, xextent: Tuple[float, float], yextent: Tuple[float, float], zextent: Tuple[float, float]):
        """

        Args:
            xextent: extent in x
            yextent: extent in y
            zextent: extent in z, set (0., 0.) for a 2D structure

        Example:
            The following initialises this class in a range of 32 x 32 px in x and y and +/- 750nm in z.
            >>> prior_struct = RandomStructure(xextent=(-0.5, 31.5), yextent=(-0.5, 31.5), zextent=(-750., 750.))

        """
        super().__init__()
        self.xextent = xextent
        self.yextent = yextent
        self.zextent = zextent

        self.scale = torch.tensor([(self.xextent[1] - self.xextent[0]),
                                   (self.yextent[1] - self.yextent[0]),
                                   (self.zextent[1] - self.zextent[0])])

        self.shift = torch.tensor([self.xextent[0],
                                   self.yextent[0],
                                   self.zextent[0]])

    @property
    def area(self) -> float:
        return (self.xextent[1] - self.xextent[0]) * (self.yextent[1] - self.yextent[0])

    def sample(self, n: int) -> torch.Tensor:
        xyz = torch.rand((n, 3)) * self.scale + self.shift
        return xyz

    @classmethod
    def parse(cls, param):
        return cls(xextent=param.Simulation.emitter_extent[0],
                   yextent=param.Simulation.emitter_extent[1],
                   zextent=param.Simulation.emitter_extent[2])
