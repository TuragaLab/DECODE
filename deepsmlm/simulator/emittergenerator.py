from abc import ABC, abstractmethod  # abstract class
import os
import sys
import torch

from ..generic.emitter import EmitterSet


class EmitterGenerator(ABC):

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def generate_set(self):
        pass

    @abstractmethod
    def pop_single_frame(self):
        pass


class RandomPhysical(EmitterGenerator):

    def __init__(self, xextent, yextent, zextent, zsigma, l_exp, num_emitter, act_pd, ep_time):
        """

        :param xextent:     extent, where to place emitters
        :param yextent:
        :param zextent:
        :param num_emitter: number of samples
        :param act_pd:  probability density (dp/dt) of initial activation
        :param ep_time: exposure time in units which correspond to act_pd
        """
        super().__init__()

        self.xextent = xextent
        self.yextent = yextent
        self.zextent = zextent
        self.zsigma = zsigma
        self.l_exp = l_exp

        self.num_emitter = num_emitter
        self.act_pd = act_pd
        self.ep_time = ep_time

        self.total_set = None

    def generate_set(self):
        xyz = torch.rand(self.num_emitter, 3)

        if z_sigma is not None:
            xyz *= torch.tensor([self.xextent[1] - self.xextent[0],
                                 self.yextent[1] - self.yextent[0],
                                 self.zextent[1] - self.zextent[0]])
        else:

            xyz *= torch.tensor([self.xextent[1] - self.xextent[0],
                                 self.yextent[1] - self.yextent[0],
                                 1])
            xyz[:, 2] = (self.zextent[1] + self.zextent[0]) / 2 + torch.randn_like(xyz[:, 2]) * self.zsigma

        phot_total = torch.zeros_like(self.num_emitter).exponential_(self.l_exp)
        id = torch.arange(0, self.num_emitter)

        self.set_total = EmitterSet(xyz, phot_total, None, id)
        return self.set_total

    def activation_physics(self):
        pass