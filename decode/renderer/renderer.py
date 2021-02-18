from abc import ABC
import torch
from ..generic import emitter
import math

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import hsv_to_rgb
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.ndimage import gaussian_filter


class Renderer(ABC):
    """
    Renderer. Takes emitters and outputs a rendered image.

    """

    def __init__(self, plot_axis: tuple, xextent: tuple, yextent: tuple, px_size: float):
        super().__init__()

        self.xextent = xextent
        self.yextent = yextent
        self.px_size = px_size
        self.plot_axis = plot_axis
        
    @property
    def _npx_x(self):
        return math.ceil(se)

    def forward(self, em: emitter.EmitterSet) -> torch.Tensor:
        """
        Forward emitterset through rendering and output rendered data.

        Args:
            em: emitter set

        """
        raise NotImplementedError

    def render(self, em: emitter.EmitterSet, ax=None):
        """
        Render emitters

        Args:
            em: emitter set
            ax: plot axis

        Returns:

        """
        raise NotImplementedError


class Renderer2D(Renderer):
    """
    2D Renderer with constant gaussian.

    """

    def __init__(self, px_size, sigma_blur, plot_axis = (0,1), xextent=None, yextent=None, clip_percentile=None):
        super().__init__(plot_axis=plot_axis,xextent=xextent, yextent=yextent, px_size=px_size)

        self.sigma_blur = sigma_blur
        self.clip_percentile = clip_percentile

    def render(self, em, ax=None, cmap: str = 'gray'):

        hist = self.forward(em).numpy()

        if ax is None:
            ax = plt.gca()

        ax = ax.imshow(np.transpose(hist), cmap=cmap)  # because imshow use different ordering
        return ax

    def forward(self, em: emitter.EmitterSet) -> torch.Tensor:

        if self.xextent is None:
            self.xextent = (em.xyz_nm[:, self.plot_axis[0]].min(), em.xyz_nm[:, self.plot_axis[0]].max())
        if self.yextent is None:
            self.yextent = (em.xyz_nm[:, self.plot_axis[1]].min(), em.xyz_nm[:, self.plot_axis[1]].max())

        hist = self._hist2d(em.xyz_nm[:, self.plot_axis].numpy(), xextent=self.xextent, yextent=self.yextent, px_size=self.px_size)

        if self.clip_percentile is not None:
            hist = np.clip(hist, 0., np.percentile(hist, self.clip_percentile))

        if self.sigma_blur is not None:
            hist = gaussian_filter(hist, sigma=[self.sigma_blur / self.px_size, self.sigma_blur / self.px_size])

        return torch.from_numpy(hist)

    @staticmethod
    def _hist2d(xy: np.array, xextent, yextent, px_size) -> np.array:

        hist_bins_x = np.arange(xextent[0], xextent[1] + px_size, px_size)
        hist_bins_y = np.arange(yextent[0], yextent[1] + px_size, px_size)

        hist, _, _ = np.histogram2d(xy[:, 0], xy[:, 1], bins=(hist_bins_x, hist_bins_y))

        return hist

    
class Renderer3D(Renderer):
    """
    3D Renderer with constant gaussian.

    """

    def __init__(self, px_size, sigma_blur, plot_axis = (0,1,2), xextent=None, yextent=None, zextent=None, clip_percentile=100, gamma=1):
        super().__init__(plot_axis=plot_axis, xextent=xextent, yextent=yextent, px_size=px_size)

        self.sigma_blur = sigma_blur
        self.clip_percentile = clip_percentile
        self.gamma = gamma
        self.zextent = zextent

    def render(self, em):

        hist = self.forward(em).numpy()

        ax = plt.gca()
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size=0.25, pad=-0.25)
        colb = mpl.colorbar.ColorbarBase(cax, cmap=plt.get_cmap('hsv'), values=np.linspace(0,0.7,101), norm=mpl.colors.Normalize(0.,1.))
        colb.outline.set_visible(False)
        colb.ax.invert_yaxis()

        cax.text(0.12, 0.04, f'{self.zextent[0]} nm', rotation=90, color='white', fontsize=15, transform=cax.transAxes)
        cax.text(0.12, 0.88, f'{self.zextent[1]} nm', rotation=90, color='white', fontsize=15, transform=cax.transAxes)
        cax.axis('off')

        ax = ax.imshow(np.transpose(hist,[1,0,2]))
        return ax

    def forward(self, em: emitter.EmitterSet) -> torch.Tensor:

        if self.xextent is None:
            self.xextent = (em.xyz_nm[:, self.plot_axis[0]].min(), em.xyz_nm[:, self.plot_axis[0]].max())
        if self.yextent is None:
            self.yextent = (em.xyz_nm[:, self.plot_axis[1]].min(), em.xyz_nm[:, self.plot_axis[1]].max())
        if self.zextent is None:
            self.zextent = (em.xyz_nm[:, self.plot_axis[2]].min(), em.xyz_nm[:, self.plot_axis[2]].max())

        int_hist, col_hist = self._hist2d(em.xyz_nm[:, self.plot_axis].numpy(), xextent=self.xextent, yextent=self.yextent, zextent=self.zextent, px_size=self.px_size)
        with np.errstate(divide='ignore', invalid='ignore'):
            z_avg = col_hist / int_hist
        
        if self.clip_percentile is not None:
            int_hist = np.clip(int_hist, 0., np.percentile(int_hist, self.clip_percentile))
        z_avg[np.isnan(z_avg)] = 0
            
        val = (int_hist - int_hist.min()) / (int_hist.max() - int_hist.min())
        sat = np.ones(int_hist.shape)
        # Revert coloraxis to be closer to the paper figures
        hue = -(z_avg * 0.65) + 0.65

        HSV = np.concatenate((hue[:, :, None], sat[:, :, None], val[:, :, None]), -1)
        RGB = hsv_to_rgb(HSV) ** (1 / self.gamma)

        if self.sigma_blur:
            RGB = np.array([gaussian_filter(RGB[:, :, i], sigma=[self.sigma_blur / self.px_size,
                                                                 self.sigma_blur / self.px_size]) for i in range(3)]).transpose(1, 2, 0)

        return torch.from_numpy(RGB)

    @staticmethod
    def _hist2d(xyz: np.array, xextent, yextent, zextent, px_size) -> np.array:
        
        hist_bins_x = np.arange(xextent[0], xextent[1] + px_size, px_size)
        hist_bins_y = np.arange(yextent[0], yextent[1] + px_size, px_size)

        int_hist, _, _ = np.histogram2d(xyz[:, 0], xyz[:, 1], bins=(hist_bins_x, hist_bins_y))
        
        z_pos = np.clip(xyz[:,2], zextent[0], zextent[1])
        z_weight = ((z_pos - z_pos.min()) / (z_pos.max() - z_pos.min()))
        
        col_hist, _, _ = np.histogram2d(xyz[:, 0], xyz[:, 1], bins=(hist_bins_x, hist_bins_y), weights=z_weight)
        
        return int_hist, col_hist