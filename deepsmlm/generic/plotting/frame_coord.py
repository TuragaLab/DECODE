import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
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
    def __init__(self, frame, extent=None, norm=None, clim=None):
        """
        :param frame: torch tensor.
        """

        self.frame = frame.detach().squeeze()
        self.extent = extent
        self.norm = LogNorm() if norm is 'log' else None
        self.clim = clim

    def plot(self):
        """
        Plot the frame. Note that according to convention we need to transpose the last two axis.
        """
        if self.extent is None:
            plt.imshow(self.frame.transpose(-1, -2).numpy(), cmap='gray', norm=self.norm)
        else:
            plt.imshow(self.frame.transpose(-1, -2).numpy(), cmap='gray', norm=self.norm, extent=(self.extent[0][0],
                                                                                                  self.extent[0][1],
                                                                                                  self.extent[1][1],
                                                                                                  self.extent[1][0]))
        plt.gca().set_aspect('equal', adjustable='box')
        if self.clim is not None:
            plt.clim(self.clim[0], self.clim[1])
            # safety measure
            # plt.colorbar()
        plt.xlabel('x')
        plt.ylabel('y')

        return plt.gca()


class PlotCoordinates:
    def __init__(self,
                 pos_tar=None, phot_tar = None,
                 pos_out=None, phot_out=None,
                 pos_ini=None, phot_ini=None,
                 extent_limit=None):
        """
        :param pos_tar: torch tensor of target values
        :param pos_pred: torch tensor of outputted pos
        :param pos_ini: torch tensor of initilaised pos
        """

        self.extent_limit = extent_limit
        self.pos_tar = pos_tar
        self.phot_tar = phot_tar
        self.pos_out = pos_out
        self.phot_out = phot_out
        self.pos_ini = pos_ini
        self.phot_ini = phot_ini

        self.tar_marker = 'ro'
        self.tar_cmap = 'winter'
        self.out_marker = 'bx'
        self.out_cmap = 'viridis'
        self.ini_marker = 'g+'
        self.ini_cmap = 'copper'

    def plot(self):
        """
        Plot the coordinates.
        """
        def plot_xyz(pos, marker, color, label):
            plt.scatter(pos[:, 0].numpy(), pos[:, 1].numpy(),
                        marker=marker, c=color, facecolors='none', label=label)

        def plot_xyz_phot(pos, phot, marker, cmap, label):
            plt.scatter(pos[:, 0].numpy(), pos[:, 1].numpy(), c=phot.numpy(),
                     marker=marker, facecolors='none', cmap=cmap ,label=label)

        if self.pos_tar is not None:
            if self.phot_tar is not None:
                plot_xyz_phot(self.pos_tar, self.phot_tar, self.tar_marker[1], self.tar_cmap, 'Target')
            else:
                plot_xyz(self.pos_tar, self.tar_marker[1], self.tar_marker[0], 'Target')

        if self.pos_out is not None:
            if self.phot_out is not None:
                plot_xyz_phot(self.pos_out, self.phot_out, self.out_marker[1], self.out_cmap, 'Output')
            else:
                plot_xyz(self.pos_out, self.out_marker[1], self.out_marker[0], 'Output')

        if self.pos_ini is not None:
            if self.phot_ini is not None:
                plot_xyz_phot(self.pos_ini, self.phot_ini, self.ini_marker[1], self.ini_cmap, 'Init')
            else:
                plot_xyz(self.pos_ini, self.ini_marker[1], self.ini_marker[0], 'Init')

        plt.gca().set_aspect('equal', adjustable='box')
        plt.xlabel('x')
        plt.ylabel('y')
        if self.extent_limit is not None:
            plt.xlim(*self.extent_limit[0])
            plt.ylim(*self.extent_limit[1][::-1])  # reverse tuple order

        return plt.gca()

class PlotCoordinates3D:
    def __init__(self, pos_tar=None, pos_out=None, phot_out=None):

        self.pos_tar = pos_tar
        self.pos_out = pos_out
        self.phot_out = phot_out

        self.fig = plt.gcf()
        self.ax = self.fig.add_subplot(111, projection='3d')

    def plot(self):
        if self.pos_tar is not None:
            xyz = self.pos_tar
            self.ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c='r', marker='o', label='Target')

        if self.pos_out is not None:
            xyz = self.pos_out
            phot = self.phot_out

            rgba_colors = torch.zeros((xyz.shape[0], 4))
            rgba_colors[:, 2] = 1.0
            rgba_colors[:, 3] = phot / phot.max()
            self.ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], marker='^', color=rgba_colors, label='Output')
            plt.xlabel('x')
            plt.ylabel('y')
            plt.gca().invert_yaxis()


class PlotFrameCoord(PlotCoordinates, PlotFrame):
    """Combination of Frame and Coord"""

    def __init__(self, frame,
                 pos_tar=None, phot_tar = None,
                 pos_out=None, phot_out=None,
                 pos_ini=None, phot_ini=None,
                 extent=None, coord_limit=None,
                 norm=None, clim=None):
        """
        (see base classes)
        :param frame:
        :param pos_tar:
        :param pos_out:
        :param pos_ini:
        """

        PlotCoordinates.__init__(self,
                                 pos_tar=pos_tar,
                                 phot_tar=phot_tar,
                                 pos_out=pos_out,
                                 phot_out=phot_out,
                                 pos_ini=pos_ini,
                                 phot_ini=phot_ini,
                                 extent_limit=coord_limit)

        PlotFrame.__init__(self, frame, extent, norm, clim)

    def plot(self):
        """
        Plot both frame and coordinates. Plot frame first because if we have emtiters outside
        the box we wont see
        them wenn we limit the extent first.
        :return: None
        """

        PlotFrame.plot(self)
        PlotCoordinates.plot(self)

if __name__ == '__main__':
    extent = ((-0.5, 31.5), (-0.5, 31.5), None)
    img_shape = (32, 32)

    img = torch.rand((img_shape[0], img_shape[1]))
    # PlotFrame(frame=img, extent=extent).plot()
    # plt.show()

    xyz = torch.rand((5, 3)) * img_shape[0]
    phot = torch.rand((5,)) * 1000

    xyz_out = torch.cat((xyz, xyz), 0)
    xyz_out += 5 * torch.randn_like(xyz_out)
    phot_out = torch.cat((phot, phot), dim=0)

    PlotCoordinates(pos_ini=xyz, phot_ini=phot).plot()
    plt.show()

    PlotFrameCoord(frame=img, pos_out=xyz, phot_out=phot).plot()
    plt.show()

    PlotCoordinates3D(pos_tar=xyz, pos_out=xyz_out, phot_out=phot_out).plot()
    plt.show()

    img = torch.zeros((25, 25)) + 0.0001
    img[10, 10] = 1
    img[15, 15] = 10
    img[20, 20] = 100
    PlotFrame(img, norm='log', clim=(0.01, 100)).plot()
    plt.show()
