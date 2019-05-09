import torch


class OffsetRescale:
    def __init__(self, scale_x: float, scale_y: float, scale_z: float, scale_phot: float, buffer=1.):
        """
        Assumes scale_x, scale_y, scale_z to be symmetric ranged, scale_phot, ranged between 0-1
        :param scale_x:
        :param scale_y:
        :param scale_z:
        :param scale_phot:
        :param buffer: to extend the original range a little bit, to use the more linear parts of a sigmoidal fct.
        Does not apply to probability channel 0.
        """

        self.sc_x = scale_x
        self.sc_y = scale_y
        self.sc_z = scale_z
        self.sc_phot = scale_phot
        self.buffer = buffer

    def forward(self, x):
        """
        Scale the NN output to the apropriate scale
        :param x: (torch.tensor, N x 5 x H x W) or 5 x H x W
        :return:
        """
        if x.dim() == 3:
            x.unsqueeze_(0)
            squeeze_before_return = True
        else:
            squeeze_before_return = False

        x_ = x.clone()

        x_[:, 1, :, :] *= self.sc_phot * self.buffer
        x_[:, 2, :, :] *= self.sc_x * self.buffer
        x_[:, 3, :, :] *= self.sc_y * self.buffer
        x_[:, 4, :, :] *= self.sc_z * self.buffer

        if squeeze_before_return:
            return x_.squeeze(0)
        else:
            return x_


class InverseOffsetRescale(OffsetRescale):
    def __init__(self, scale_x: float, scale_y: float, scale_z: float, scale_phot: float, buffer=1.):
        """
        Assumes scale_x, scale_y, scale_z to be symmetric ranged, scale_phot, ranged between 0-1
        :param scale_x:
        :param scale_y:
        :param scale_z:
        :param scale_phot:
        """
        super().__init__(scale_x, scale_y, scale_z, scale_phot, buffer)

    def forward(self, x):
        """
        Scale the original output to the NN range
        :param x:
        :return:
        """
        if x.dim() == 3:
            x.unsqueeze_(0)
            squeeze_before_return = True
        else:
            squeeze_before_return = False

        x_ = x.clone()

        x_[:, 1, :, :] /= (self.sc_phot * self.buffer)
        x_[:, 2, :, :] /= (self.sc_x * self.buffer)
        x_[:, 3, :, :] /= (self.sc_y * self.buffer)
        x_[:, 4, :, :] /= (self.sc_z * self.buffer)

        if squeeze_before_return:
            return x_.squeeze(0)
        else:
            return x_
