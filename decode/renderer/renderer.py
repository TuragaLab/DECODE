from abc import ABC
from typing import Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import hsv_to_rgb
from matplotlib.colors import rgb_to_hsv
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.ndimage import gaussian_filter
from tqdm import tqdm

from ..generic import emitter


class Renderer(ABC):
    def __init__(self, plot_axis: tuple, xextent: tuple, yextent: tuple, zextent: tuple,
                 px_size: float, abs_clip: float, rel_clip: float, contrast: float):
        """Renderer. Takes emitters and outputs a rendered image."""
        super().__init__()

        self.xextent = xextent
        self.yextent = yextent
        self.zextent = zextent

        self.px_size = px_size
        self.plot_axis = plot_axis

        self.abs_clip = abs_clip
        self.rel_clip = rel_clip

        self.contrast = contrast

        assert (
                self.abs_clip is None or self.rel_clip is None
        ), "Define either an absolute or a relative value for clipping, but not both"

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

        """
        raise NotImplementedError


class Renderer2D(Renderer):
    def __init__(self, px_size, sigma_blur, plot_axis=(0, 1, 2),
                 xextent=None, yextent=None, zextent=None, colextent=None,
                 abs_clip=None, rel_clip=None, contrast=1):
        """
        2D histogram renderer with constant gaussian blur.

        Args:
            px_size: pixel size of the output image in nm
            sigma_blur: sigma of the gaussian blur applied in nm
            plot_axis: determines which dimensions get plotted. 0,1,2 = x,y,z. (0,1) is x over y.
            xextent: extent in x in nm
            yextent: extent in y in nm
            zextent: extent in z in nm.
            cextent: extent of the color variable. Values outside of this range get clipped.
            abs_clip: absolute clipping value of the histogram in counts
            rel_clip: clipping value relative to the maximum count. i.e. rel_clip = 0.8 clips at 0.8*hist.max()
            contrast: scaling factor to increase contrast

        """
        super().__init__(
            plot_axis=plot_axis,
            xextent=xextent,
            yextent=yextent,
            zextent=zextent,
            px_size=px_size,
            abs_clip=abs_clip,
            rel_clip=rel_clip,
            contrast=contrast,
        )

        self.sigma_blur = sigma_blur
        self.colextent = colextent

        self.jet_hue = self._get_jet_cmap()

    def render(self, em: emitter.EmitterSet, col_vec=None, ax=None):
        """
        Forward emitterset through rendering and output rendered data.

        Args:
            em: emitter set
            col_vec: torch tensor (1 dim) with the same length as em
            ax: plot axis

        """
        hist = self.forward(em, col_vec).numpy()

        if ax is None:
            ax = plt.gca()

        if col_vec is not None:

            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size=0.25, pad=-0.25)
            colb = mpl.colorbar.ColorbarBase(
                cax, cmap=plt.get_cmap("jet"), values=np.linspace(0, 1.0, 101),
                norm=mpl.colors.Normalize(0.0, 1.0)
            )
            colb.outline.set_visible(False)

            cax.text(
                0.12, 0.04, f"{self.colextent[0]}", rotation=90, color="white", fontsize=15,
                transform=cax.transAxes
            )
            cax.text(
                0.12, 0.88, f"{self.colextent[1]}", rotation=90, color="white", fontsize=15,
                transform=cax.transAxes
            )
            cax.axis("off")

            ax.imshow(np.transpose(hist, [1, 0, 2]))

        else:

            # because imshow use different ordering
            ax.imshow(np.transpose(hist), cmap="gray")

        return ax

    def forward(self, em: emitter.EmitterSet, col_vec=None) -> torch.Tensor:
        """
        Forward emitterset through rendering and output rendered data.

        Args:
            em: emitter set
            col_vec: torch tensor (1 dim) with the same length as em

        """

        xyz_extent = self.get_extent(em)
        ind_mask = (
                (em.xyz_nm[:, 0] >= xyz_extent[0][0])
                * (em.xyz_nm[:, 0] <= xyz_extent[0][1])
                * (em.xyz_nm[:, 1] >= xyz_extent[1][0])
                * (em.xyz_nm[:, 1] <= xyz_extent[1][1])
                * (em.xyz_nm[:, 2] >= xyz_extent[2][0])
                * (em.xyz_nm[:, 2] <= xyz_extent[2][1])
        )

        em_sub = em[ind_mask]

        if col_vec is not None:

            col_vec = col_vec[ind_mask]
            self.colextent = (
            col_vec.min(), col_vec.max()) if self.colextent is None else self.colextent
            int_hist, col_hist = self._hist2d(
                em_sub, col_vec, xyz_extent[self.plot_axis[0]], xyz_extent[self.plot_axis[1]],
                self.colextent
            )

            with np.errstate(divide="ignore", invalid="ignore"):
                c_avg = col_hist / int_hist

            if self.rel_clip is not None:
                int_hist = np.clip(int_hist * self.contrast, 0.0, int_hist.max() * self.rel_clip)
                val = int_hist / int_hist.max()
            elif self.abs_clip is not None:
                int_hist = np.clip(int_hist, 0.0, self.abs_clip)
                val = int_hist / self.abs_clip
            else:
                val = int_hist / int_hist.max()

            val *= self.contrast

            c_avg[np.isnan(c_avg)] = 0
            sat = np.ones(int_hist.shape)
            hue = np.interp(c_avg, np.linspace(0, 1, 256), self.jet_hue)

            HSV = np.concatenate((hue[:, :, None], sat[:, :, None], val[:, :, None]), -1)
            RGB = hsv_to_rgb(HSV)

            if self.sigma_blur:
                RGB = np.array(
                    [
                        gaussian_filter(
                            RGB[:, :, i],
                            sigma=[self.sigma_blur / self.px_size, self.sigma_blur / self.px_size]
                        )
                        for i in range(3)
                    ]
                ).transpose(1, 2, 0)

            RGB = np.clip(RGB, 0, 1)
            return torch.from_numpy(RGB)

        else:

            hist = self._hist2d(em_sub, None, xyz_extent[self.plot_axis[0]],
                                xyz_extent[self.plot_axis[1]])

            if self.rel_clip is not None:
                hist = np.clip(hist, 0.0, hist.max() * self.rel_clip)
            if self.abs_clip is not None:
                hist = np.clip(hist, 0.0, self.abs_clip)

            if self.sigma_blur is not None:
                hist = gaussian_filter(hist, sigma=[self.sigma_blur / self.px_size,
                                                    self.sigma_blur / self.px_size])

            hist = np.clip(hist, 0, hist.max() / self.contrast)
            return torch.from_numpy(hist)

    def get_extent(self, em) -> Tuple[tuple, tuple, tuple]:

        xextent = (
        em.xyz_nm[:, 0].min(), em.xyz_nm[:, 0].max()) if self.xextent is None else self.xextent
        yextent = (
        em.xyz_nm[:, 1].min(), em.xyz_nm[:, 1].max()) if self.yextent is None else self.yextent
        zextent = (
        em.xyz_nm[:, 2].min(), em.xyz_nm[:, 2].max()) if self.zextent is None else self.zextent

        return xextent, yextent, zextent

    def _hist2d(self, em: emitter.EmitterSet, col_vec, x_hist_ext, y_hist_ext, c_range=None):

        xy = em.xyz_nm[:, self.plot_axis].numpy()

        hist_bins_x = np.arange(x_hist_ext[0], x_hist_ext[1] + self.px_size, self.px_size)
        hist_bins_y = np.arange(y_hist_ext[0], y_hist_ext[1] + self.px_size, self.px_size)

        int_hist, _, _ = np.histogram2d(xy[:, 0], xy[:, 1], bins=(hist_bins_x, hist_bins_y))

        if col_vec is not None:

            c_pos = np.clip(col_vec, c_range[0], c_range[1])
            c_weight = (c_pos - c_range[0]) / (c_range[1] - c_range[0])

            col_hist, _, _ = np.histogram2d(xy[:, 0], xy[:, 1], bins=(hist_bins_x, hist_bins_y),
                                            weights=c_weight)

            return int_hist, col_hist

        else:

            return int_hist

    @staticmethod
    def _get_jet_cmap():
        lin_hue = np.linspace(0, 1, 256)
        cmap = plt.get_cmap("jet", lut=256)
        cmap = cmap(lin_hue)
        cmap_hsv = rgb_to_hsv(cmap[:, :3])
        jet_hue = cmap_hsv[:, 0]
        _, b = np.unique(jet_hue, return_index=True)
        jet_hue = [jet_hue[index] for index in sorted(b)]
        jet_hue = np.interp(np.linspace(0, len(jet_hue), 256), np.arange(len(jet_hue)), jet_hue)
        return jet_hue


class RendererIndividual2D(Renderer2D):
    def __init__(self, px_size, batch_size=1000, filt_size=10, plot_axis=(0, 1),
                 xextent=None, yextent=None, zextent=None, colextent=None,
                 abs_clip=None, rel_clip=None, contrast=1, intensity_field="sigma", device="cpu"):
        """
        2D histogram renderer. Each localization is individually rendered as a 2D Gaussian corresponding to a
        respective field.

        Args:
            px_size: pixel size of the output image in nm
            batch_size: number of localization processed in parallel
            filt_size: each gaussian is calculated as a patch with size filt_size*filt_size (in pixels)
            plot_axis: determines which dimensions get plotted. 0,1,2 = x,y,z. (0,1) is x over y.
            xextent: extent in x in nm
            yextent: extent in y in nm
            zextent: extent in z in nm.
            cextent: extent of the color variable. Values outside of this range get clipped.
            abs_clip: absolute clipping value of the histogram in counts
            rel_clip: clipping value relative to the maximum count. i.e. rel_clip = 0.8 clips at 0.8*hist.max()
            contrast: scaling factor to increase contrast
            intensity_field: field of emitter that should be used for rendering
            device: render on cpu or cuda

        """
        super().__init__(
            px_size=px_size,
            sigma_blur=None,
            plot_axis=plot_axis,
            xextent=xextent,
            yextent=yextent,
            zextent=zextent,
            colextent=colextent,
            abs_clip=abs_clip,
            rel_clip=rel_clip,
            contrast=contrast,
        )

        self.bs = batch_size
        self.fs = filt_size
        self.device = device
        self.intensity_field = intensity_field

    def calc_gaussians(self, xy_mu, xy_sig, mesh):

        xy_mu = xy_mu[:, :2] % self.px_size / self.px_size
        xy_sig = xy_sig[:, :2] / self.px_size

        dist = torch.distributions.Normal(xy_mu, xy_sig)
        W = torch.exp(dist.log_prob(mesh[:, :, None]).sum(-1)).permute(2, 0, 1)

        return W / torch.clamp_min(W.sum(-1).sum(-1), 1.0)[:, None, None]

    @torch.jit.script
    def _place_gaussians(int_hist, inds, W, fs):
        for i in range(len(W)):
            int_hist[inds[i, 1]: inds[i, 1] + fs, inds[i, 0]: inds[i, 0] + fs] += W[i]

        return int_hist

    @torch.jit.script
    def _place_gaussians_weighted(comb_hist, inds, weights, W, fs):
        for i in range(len(W)):
            comb_hist[inds[i, 1]: inds[i, 1] + fs, inds[i, 0]: inds[i, 0] + fs] \
                += torch.stack([W[i], W[i] * weights[i]], -1)

        return comb_hist

    def _hist2d(self, em: emitter.EmitterSet, col_vec, x_hist_ext, y_hist_ext, c_range=None):

        ym, xm = torch.meshgrid(
            torch.linspace(-(self.fs // 2), self.fs // 2, self.fs, device=self.device),
            torch.linspace(-(self.fs // 2), self.fs // 2, self.fs, device=self.device),
        )

        mesh = torch.cat([(xm)[..., None], (ym)[..., None]], -1)

        xy_mus = em.xyz_nm[:, self.plot_axis].to(self.device)
        xy_sigs = em.xyz_sig_nm[:, self.plot_axis].to(self.device)

        w = int((x_hist_ext[1] - x_hist_ext[0]) // self.px_size + 1)
        h = int((y_hist_ext[1] - y_hist_ext[0]) // self.px_size + 1)

        s_inds = xy_mus - torch.Tensor([x_hist_ext[0], y_hist_ext[0]], device=self.device)
        s_inds = torch.div(s_inds, self.px_size, rounding_mode="trunc").long()

        if col_vec is not None:

            c_pos = torch.clip(col_vec, c_range[0], c_range[1])
            c_weight = ((c_pos - c_range[0]) / (c_range[1] - c_range[0])).to(self.device)

            comb_hist = torch.zeros([h + self.fs, w + self.fs, 2], device=self.device,
                                    dtype=torch.float)

            for i in tqdm(range(len(xy_mus) // self.bs + 1)):
                sl = np.s_[i * self.bs: (i + 1) * self.bs]
                sub_inds = s_inds[sl]
                W = self.calc_gaussians(xy_mus[sl], xy_sigs[sl], mesh)
                c_ws = c_weight[sl]
                comb_hist = self._place_gaussians_weighted(comb_hist, sub_inds, c_ws, W,
                                                           torch.tensor(self.fs))

            comb_hist = comb_hist[self.fs // 2: -(self.fs // 2 + 1),
                        self.fs // 2: -(self.fs // 2 + 1)]
            int_hist = comb_hist[:, :, 0]
            col_hist = comb_hist[:, :, 1]

            return int_hist.T.cpu().numpy(), col_hist.T.cpu().numpy()

        else:

            int_hist = torch.zeros([h + self.fs, w + self.fs], device=self.device,
                                   dtype=torch.float)

            for i in tqdm(range(len(xy_mus) // self.bs + 1)):
                sl = np.s_[i * self.bs: (i + 1) * self.bs]
                sub_inds = s_inds[sl]
                W = self.calc_gaussians(xy_mus[sl], xy_sigs[sl], mesh)
                int_hist = self._place_gaussians(int_hist, sub_inds, W, torch.tensor(self.fs))

            int_hist = int_hist[self.fs // 2: -(self.fs // 2 + 1),
                       self.fs // 2: -(self.fs // 2 + 1)]

            return int_hist.T.cpu().numpy()
