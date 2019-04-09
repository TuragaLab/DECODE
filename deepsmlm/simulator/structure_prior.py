import torch

from abc import ABC, abstractmethod


class StructurePrior(ABC):

    @abstractmethod
    def __init__(self):
        """
        Describe the probabilistic model of the structure here.
        """
        pass

    @abstractmethod
    def get_area(self):
        """
        Calculate the area which is occupied by the structure. This is useful to later calculate the density,
        and the effective number of emitters).

        :return:
        """
        pass

    @abstractmethod
    def draw(self, n, dim=3):
        """
        Draw n sample positions from structure

        :param n: (int) number of samples
        :param dim: (int) 3 for xyz, 2 for xyz with z=0
        :return: (torch.tensor) corodinates
        """
        pass


class RandomStructure(StructurePrior):

    def __init__(self, xextent, yextent, zextent):
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

    def get_area(self):
        return (self.xextent[1] - self.xextent[0]) * (self.yextent[1] - self.yextent[0])

    def draw(self, n, dim=3):
        xyz = torch.rand((n, 3)) * self.scale + self.shift
        if dim == 2:
            xyz[:, 2] = 0.
        return xyz


class DiscreteZStructure(StructurePrior):

    def __init__(self, xy_pos, z_abs_max, eps=10):
        super().__init__()
        self.xy_pos = xy_pos
        self.z_abs_max = z_abs_max
        self.eps = eps

    def get_area(self):
        return None

    def draw(self, n, dim=3):
        xyz = torch.ones((n, dim)) * torch.cat((self.xy_pos, torch.tensor([0.])), 0)
        z_ix = torch.randint(-1, 1+1, (n,), dtype=torch.float)
        z_value = z_ix * self.z_abs_max + torch.randn_like(z_ix) * self.eps
        xyz[:, 2] = z_value

        return xyz


if __name__ == '__main__':
    zs = DiscreteZStructure(torch.tensor([15., 15.]), 500, eps=10)
    xyz = zs.draw(10, 3)
    print(xyz)

    print("Pseudotest done.")