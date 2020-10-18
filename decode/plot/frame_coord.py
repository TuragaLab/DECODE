import matplotlib.pyplot as plt
import torch
from matplotlib.colors import LogNorm

"""
Convention:
x to the right, y down.
—--—x --——> 
y
|
v
"""


def connect_point_set(set0, set1, threeD=False, ax=None):
    """
    Plots the connecting lines between the set0 and set1 in 2D.

    Args:
        set0:  torch.Tensor / np.array of dim N x 2
        set1:  torch.Tensor / np.array of dim N x 2
        threeD (bool): plot / connect in 3D
        ax:  axis where to plot

    Returns:

    """
    if ax is None:
        ax = plt.gca()

    if threeD:
        for i in range(set0.size(0)):
            ax.plot3D([set0[i, 0], set1[i, 0]], [set0[i, 1], set1[i, 1]], [set0[i, 2], set1[i, 2]], 'orange')
    else:
        for i in range(set0.size(0)):
            ax.plot([set0[i, 0], set1[i, 0]], [set0[i, 1], set1[i, 1]], 'orange')


class PlotFrame:
    def __init__(self, frame, extent=None, norm=None, clim=None, plot_colorbar=False):
        """
        :param frame: torch tensor.
        """

        self.frame = frame.detach().squeeze()
        self.extent = extent
        self.norm = LogNorm() if norm == 'log' else None
        self.clim = clim
        self.plot_colorbar = plot_colorbar

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
        if self.plot_colorbar:
            plt.colorbar(fraction=0.046, pad=0.04)
        plt.xlabel('x')
        plt.ylabel('y')

        return plt.gca()


class PlotCoordinates:
    _labels_default = ('Target', 'Output', 'Init')

    def __init__(self,
                 pos_tar=None, phot_tar=None,
                 pos_out=None, phot_out=None,
                 pos_ini=None, phot_ini=None,
                 extent_limit=None,
                 match_lines=False,
                 labels=None):
        """
        Plots points in 2D projection.
        Args:
            pos_tar:
            phot_tar:
            pos_out:
            phot_out:
            pos_ini:
            phot_ini:
            extent_limit:
            match_lines: plots
        """

        self.extent_limit = extent_limit
        self.pos_tar = pos_tar
        self.phot_tar = phot_tar
        self.pos_out = pos_out
        self.phot_out = phot_out
        self.pos_ini = pos_ini
        self.phot_ini = phot_ini
        self.match_lines = match_lines
        self.labels = labels if labels is not None else self._labels_default

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
                        marker=marker, facecolors='none', cmap=cmap, label=label)

        if self.pos_tar is not None:
            if self.phot_tar is not None:
                plot_xyz_phot(self.pos_tar, self.phot_tar, self.tar_marker[1], self.tar_cmap, self.labels[0])
            else:
                plot_xyz(self.pos_tar, self.tar_marker[1], self.tar_marker[0], self.labels[0])

        if self.pos_out is not None:
            if self.phot_out is not None:
                plot_xyz_phot(self.pos_out, self.phot_out, self.out_marker[1], self.out_cmap, self.labels[1])
            else:
                plot_xyz(self.pos_out, self.out_marker[1], self.out_marker[0], self.labels[1])

        if self.pos_ini is not None:
            if self.phot_ini is not None:
                plot_xyz_phot(self.pos_ini, self.phot_ini, self.ini_marker[1], self.ini_cmap, self.labels[2])
            else:
                plot_xyz(self.pos_ini, self.ini_marker[1], self.ini_marker[0], self.labels[2])

        if self.pos_tar is not None and self.pos_out is not None and self.match_lines:
            connect_point_set(self.pos_tar, self.pos_out, threeD=False)

        plt.gca().set_aspect('equal', adjustable='box')
        plt.xlabel('x')
        plt.ylabel('y')
        if self.extent_limit is not None:
            plt.xlim(*self.extent_limit[0])
            plt.ylim(*self.extent_limit[1][::-1])  # reverse tuple order

        return plt.gca()


class PlotCoordinates3D:
    _labels_default = ('Target', 'Output', 'Init')

    def __init__(self, pos_tar=None, pos_out=None, phot_out=None, match_lines=False, labels=None):

        self.pos_tar = pos_tar
        self.pos_out = pos_out
        self.phot_out = phot_out

        self.match_lines = match_lines
        self.labels = labels if labels is not None else self._labels_default

        self.fig = plt.gcf()
        self.ax = self.fig.add_subplot(111, projection='3d')

    def plot(self):
        if self.pos_tar is not None:
            xyz = self.pos_tar
            self.ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c='red', marker='o', label=self.labels[0])

        if self.pos_out is not None:
            xyz = self.pos_out

            rgba_colors = torch.zeros((xyz.shape[0], 4))
            rgba_colors[:, 2] = 1.0
            rgba_colors[:, 3] = 1.0
            self.ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], marker='^', color=rgba_colors.numpy(), label=self.labels[1])
            plt.xlabel('x')
            plt.ylabel('y')
            plt.gca().invert_yaxis()

        if self.pos_tar is not None and self.pos_out is not None and self.match_lines:
            connect_point_set(self.pos_tar, self.pos_out, threeD=True)


class PlotFrameCoord(PlotCoordinates, PlotFrame):
    """Combination of Frame and Coord"""

    def __init__(self, frame,
                 pos_tar=None, phot_tar=None,
                 pos_out=None, phot_out=None,
                 pos_ini=None, phot_ini=None,
                 extent=None, coord_limit=None,
                 norm=None, clim=None,
                 match_lines=False, labels=None,
                 plot_colorbar_frame=False):
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
                                 extent_limit=coord_limit,
                                 match_lines=match_lines,
                                 labels=labels)

        PlotFrame.__init__(self, frame, extent, norm, clim, plot_colorbar=plot_colorbar_frame)

    def plot(self):
        """
        Plot both frame and coordinates. Plot frame first because if we have emtiters outside
        the box we wont see
        them wenn we limit the extent first.
        :return: None
        """

        PlotFrame.plot(self)
        PlotCoordinates.plot(self)
