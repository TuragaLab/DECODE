from abc import ABC, abstractmethod  # abstract class
import torch

from deepsmlm.generic.emitter import EmitterSet


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
        frame_ix = torch.ones_like(phot_total) * float('NaN')
        id = torch.arange(0, self.num_emitter)

        self.set_total = EmitterSet(xyz=xyz,
                                    phot=phot_total,
                                    frame_ix=frame_ix,
                                    id=id)
        return self.set_total

    def activation_physics(self, ):
        pass


def emitters_from_csv(csv_file, img_size, cont_radius=3):
    emitters_matlab = pd.read_csv(csv_file)

    emitters_matlab = torch.from_numpy(emitters_matlab.iloc[:, :].as_matrix()).type(torch.float32)
    em_mat = torch.cat((emitters_matlab[:, 2:5] * (img_size[0] + 2 * cont_radius) - cont_radius,  # transform from 0, 1
                        emitters_matlab[:, 5:6],
                        emitters_matlab[:, 1:2] - 1,  # index shift from matlab to python
                        torch.zeros_like(emitters_matlab[:, 0:1])), dim=1)

    warnings.warn('Emitter ID not implemented yet.')

    return split_emitter_cont(em_mat, img_size)


def split_emitter_cont(em_mat, img_size):
    """

    :param em_mat: matrix of all emitters
    :param img_size: img_size in px (not upscaled)
    :return: emitter_matrix and contaminator matrix. contaminators are emitters which are outside the image
    """

    if img_size[0] != img_size[1]:
        raise NotImplementedError("Image must be square at the moment because otherwise the following doesn't work.")

    is_emit = torch.mul((em_mat[:, :2] >= 0).all(1), (em_mat[:, :2] <= img_size[0] - 1).all(1))
    is_cont = ~is_emit

    emit_mat, cont_mat = em_mat[is_emit, :], em_mat[is_cont, :]
    return emit_mat, cont_mat


# https://discuss.pytorch.org/t/efficient-distance-matrix-computation/9065
def pairwise_distances(x, y=None):  # not numerically stable but fast
    '''
    Input: x is a Nxd matrix
           y is an optional Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
            if y is not given then use 'y=x'.
    i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
    '''
    x_norm = (x**2).sum(1).view(-1, 1)
    if y is not None:
        y_norm = (y**2).sum(1).view(1, -1)
    else:
        y = x
        y_norm = x_norm.view(1, -1)

    dist = x_norm + y_norm - 2.0 * torch.mm(x, torch.transpose(y, 0, 1))
    return dist