import matplotlib.pyplot as plt
import torch

"""
Convention:
x to the right, y down.
—--—x --——> 
y
|
v
"""


class PlotFrame:
    def __init__(self, frame, extent=None):
        """
        :param frame: torch tensor.
        """

        self.frame = frame.detach().squeeze()
        self.extent = extent

    def plot(self):
        """
        Plot the frame. Note that according to convention we need to transpose the last two axis.
        """
        if self.extent is None:
            plt.imshow(self.frame.transpose(-1, -2).numpy(), cmap='gray')
        else:
            plt.imshow(self.frame.transpose(-1, -2).numpy(), cmap='gray', extent=(self.extent[0][0],
                                                                                  self.extent[0][1],
                                                                                  self.extent[1][1],
                                                                                  self.extent[1][0]))
        plt.axis('equal')
        plt.xlabel('x')
        plt.ylabel('y')


class PlotCoordinates:
    def __init__(self, pos_tar=None, pos_out=None, pos_ini=None, extent_limit=None):
        """
        :param pos_tar: torch tensor of target values
        :param pos_pred: torch tensor of outputted pos
        :param pos_ini: torch tensor of initilaised pos
        """

        self.extent_limit = extent_limit
        self.pos_tar = pos_tar
        self.pos_out = pos_out
        self.pos_ini = pos_ini

    def plot(self):
        """
        Plot the coordinates.
        """
        if self.pos_tar is not None:
            plt.plot(self.pos_tar[:, 0].numpy(), self.pos_tar[:, 1].numpy(),
                     'ro', fillstyle='none', label='Target')
        if self.pos_out is not None:
            plt.plot(self.pos_out[:, 0].numpy(), self.pos_out[:, 1].numpy(),
                     'bx', fillstyle='none', label='Output')
        if self.pos_ini is not None:
            plt.plot(self.pos_ini[:, 0].numpy(), self.pos_ini[:, 1].numpy(),
                     'g+', fillstyle='none', label='Init')

        plt.axis('equal')
        plt.xlabel('x')
        plt.ylabel('y')
        if self.extent_limit is not None:
            plt.xlim(*self.extent_limit[0])
            plt.ylim(*self.extent_limit[1][::-1])  # reverse tuple order


class PlotFrameCoord(PlotCoordinates, PlotFrame):
    """Combination of Frame and Coord"""

    def __init__(self, frame, pos_tar=None, pos_out=None, pos_ini=None, extent=None, coord_limit=None):
        """
        (see base classes)
        :param frame:
        :param pos_tar:
        :param pos_out:
        :param pos_ini:
        """
        PlotCoordinates.__init__(self, pos_tar=pos_tar, pos_out=pos_out, pos_ini=pos_ini,
                                 extent_limit=coord_limit)
        PlotFrame.__init__(self, frame, extent)

    def plot(self):
        """
        Plot both frame and coordinates. Plot frame first because if we have emtiters outside
        the box we wont see
        them wenn we limit the extent first.
        :return: None
        """
        PlotFrame.plot(self)
        PlotCoordinates.plot(self)
