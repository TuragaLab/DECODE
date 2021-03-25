from typing import Optional

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
            ax.plot3D([set0[i, 0], set1[i, 0]], [set0[i, 1], set1[i, 1]], [set0[i, 2], set1[i, 2]],
                      'orange')
    else:
        for i in range(set0.size(0)):
            ax.plot([set0[i, 0], set1[i, 0]], [set0[i, 1], set1[i, 1]], 'orange')


class PlotFrame:
    def __init__(self, frame: torch.Tensor, extent: Optional[tuple] = None, clim=None,
                 plot_colorbar: bool = False, axes_order: Optional[str] = None):
        """
        Plots a frame.

        Args:
            frame: frame to be plotted
            extent: specify frame extent, tuple ((x0, x1), (y0, y1))
            clim: clim values
            plot_colorbar: plot the colorbar
            axes_order: order of axis. Either default order (None) or 'future'
             (i.e. future version of decode in which we will swap axes).
             This is only a visual effect and does not change the storage scheme of the EmitterSet


        """

        self.frame = frame.detach().squeeze()
        self.extent = extent
        self.clim = clim
        self.plot_colorbar = plot_colorbar
        self._axes_order = axes_order

        assert self._axes_order is None or self._axes_order == 'future'

        if self._axes_order is None:
            self.frame.transpose_(-1, -2)

    def plot(self) -> plt.axis:
        """
        Plot the frame. Note that according to convention we need to transpose the last two axis.
        """
        if self.extent is None:
            plt.imshow(self.frame.numpy(), cmap='gray')
        else:
            plt.imshow(self.frame.numpy(), cmap='gray', extent=(
                self.extent[0][0],
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
                 labels=None,
                 axes_order: Optional[str] = None):
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
            axes_order: order of axis. Either default order (None) or 'future'
             (i.e. future version of decode in which we will swap axes).
             This is only a visual effect and does not change the storage scheme of the EmitterSet

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
        self._axes_order = axes_order

        self.tar_marker = 'ro'
        self.tar_cmap = 'winter'
        self.out_marker = 'bx'
        self.out_cmap = 'viridis'
        self.ini_marker = 'g+'
        self.ini_cmap = 'copper'

        assert self._axes_order is None or self._axes_order == 'future'

    def plot(self):

        def plot_xyz(pos, marker, color, label):
            if self._axes_order == 'future':
                pos = pos[:, [1, 0, 2]]

            plt.scatter(pos[:, 0].numpy(), pos[:, 1].numpy(),
                        marker=marker, c=color, facecolors='none', label=label)

        def plot_xyz_phot(pos, phot, marker, cmap, label):
            if self._axes_order == 'decode_future':
                pos = pos[:, [1, 0, 2]]

            plt.scatter(pos[:, 0].numpy(), pos[:, 1].numpy(), c=phot.numpy(),
                        marker=marker, facecolors='none', cmap=cmap, label=label)

        if self.pos_tar is not None:
            if self.phot_tar is not None:
                plot_xyz_phot(self.pos_tar, self.phot_tar, self.tar_marker[1], self.tar_cmap,
                              self.labels[0])
            else:
                plot_xyz(self.pos_tar, self.tar_marker[1], self.tar_marker[0], self.labels[0])

        if self.pos_out is not None:
            if self.phot_out is not None:
                plot_xyz_phot(self.pos_out, self.phot_out, self.out_marker[1], self.out_cmap,
                              self.labels[1])
            else:
                plot_xyz(self.pos_out, self.out_marker[1], self.out_marker[0], self.labels[1])

        if self.pos_ini is not None:
            if self.phot_ini is not None:
                plot_xyz_phot(self.pos_ini, self.phot_ini, self.ini_marker[1], self.ini_cmap,
                              self.labels[2])
            else:
                plot_xyz(self.pos_ini, self.ini_marker[1], self.ini_marker[0], self.labels[2])

        if self.pos_tar is not None and self.pos_out is not None and self.match_lines:
            connect_point_set(self.pos_tar, self.pos_out, threeD=False)

        ax = plt.gca()
        ax.set_aspect('equal', adjustable='box')

        ax_ylimits = ax.get_ylim()
        if ax_ylimits[0] <= ax_ylimits[1]:
            ax.set_ylim(ax_ylimits[::-1])  # invert the axis

        if self._axes_order is None:
            plt.xlabel('x')
            plt.ylabel('y')
        else:
            plt.xlabel('y')
            plt.ylabel('x')

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
            self.ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c='red', marker='o',
                            label=self.labels[0])

        if self.pos_out is not None:
            xyz = self.pos_out

            rgba_colors = torch.zeros((xyz.shape[0], 4))
            rgba_colors[:, 2] = 1.0
            rgba_colors[:, 3] = 1.0
            self.ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], marker='^', color=rgba_colors.numpy(),
                            label=self.labels[1])
            plt.xlabel('x')
            plt.ylabel('y')
            plt.gca().invert_yaxis()

        if self.pos_tar is not None and self.pos_out is not None and self.match_lines:
            connect_point_set(self.pos_tar, self.pos_out, threeD=True)


class PlotFrameCoord(PlotCoordinates, PlotFrame):

    def __init__(self, frame,
                 pos_tar=None, phot_tar=None,
                 pos_out=None, phot_out=None,
                 pos_ini=None, phot_ini=None,
                 extent=None, coord_limit=None,
                 norm=None, clim=None,
                 match_lines=False, labels=None,
                 plot_colorbar_frame: bool = False,
                 axes_order: Optional[str] = None):

        PlotCoordinates.__init__(self,
                                 pos_tar=pos_tar,
                                 phot_tar=phot_tar,
                                 pos_out=pos_out,
                                 phot_out=phot_out,
                                 pos_ini=pos_ini,
                                 phot_ini=phot_ini,
                                 extent_limit=coord_limit,
                                 match_lines=match_lines,
                                 labels=labels,
                                 axes_order=axes_order)

        PlotFrame.__init__(self, frame, extent, clim,
                           plot_colorbar=plot_colorbar_frame, axes_order=axes_order)

    def plot(self):
        PlotFrame.plot(self)
        PlotCoordinates.plot(self)
