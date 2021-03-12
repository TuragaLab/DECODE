import math
from abc import ABC

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import hsv_to_rgb
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.ndimage import gaussian_filter
from matplotlib.colors import rgb_to_hsv
from ..generic import emitter

from tqdm import tqdm
from torch.jit import script

class Renderer(ABC):
    """
    Renderer. Takes emitters and outputs a rendered image.

    """

    def __init__(self, plot_axis: tuple, xextent: tuple, yextent: tuple, zextent: tuple, px_size: float, abs_clip: float, rel_clip: float, contrast: float):
        super().__init__()

        self.xextent = xextent
        self.yextent = yextent
        self.zextent = zextent
        
        self.px_size = px_size
        self.plot_axis = plot_axis
        
        self.abs_clip = abs_clip
        self.rel_clip = rel_clip
        
        self.contrast = contrast
        
        assert self.abs_clip is None or self.rel_clip is None, "Define either an absolute or a relative value for clipping, but not both"
        
    def get_extent(self, em):
        
        xextent = (em.xyz_nm[:, 0].min(), em.xyz_nm[:, 0].max()) if self.xextent is None else self.xextent
        yextent = (em.xyz_nm[:, 1].min(), em.xyz_nm[:, 1].max()) if self.yextent is None else self.yextent
        zextent = (em.xyz_nm[:, 2].min(), em.xyz_nm[:, 2].max()) if self.zextent is None else self.zextent
        
        return xextent, yextent, zextent
  
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

    Args:
        px_size: pixel size of the output image in nm
        sigma_blur: sigma of the gaussian blur applied in nm
        plot_axis: determines which dimensions get plotted. 0,1,2 = x,y,z. (0,1) is x over y, (2,1) is z over y.
        xextent: extent in x in nm
        yextent: extent in y in nm
        zextent: extent in z in nm
        abs_clip: absolute clipping value of the histogram in counts
        rel_clip: clipping value relative to the maximum count. i.e. rel_clip = 0.8 clips at 0.8*hist.max()
        contrast: scaling factor to increase contrast
    """

    def __init__(self, px_size, sigma_blur, plot_axis=(0, 1), xextent=None, yextent=None, zextent=None, abs_clip=None, rel_clip=None, contrast=1):
        super().__init__(plot_axis=plot_axis, xextent=xextent, yextent=yextent, zextent=zextent, px_size=px_size, abs_clip=abs_clip, rel_clip=rel_clip, contrast=contrast)
        
        self.sigma_blur = sigma_blur
        
    def render(self, em, ax=None, cmap: str = 'gray'):

        hist = self.forward(em).numpy()

        if ax is None:
            ax = plt.gca()

        ax.imshow(np.transpose(hist), cmap=cmap)  # because imshow use different ordering
        return ax

    def forward(self, em: emitter.EmitterSet) -> torch.Tensor:

        xyz_extent = self.get_extent(em)
        em_sub = em[(em.xyz_nm[:, 0] > xyz_extent[0][0]) * (em.xyz_nm[:, 0] < xyz_extent[0][1]) * 
                    (em.xyz_nm[:, 1] > xyz_extent[1][0]) * (em.xyz_nm[:, 1] < xyz_extent[1][1]) *
                    (em.xyz_nm[:, 2] > xyz_extent[2][0]) * (em.xyz_nm[:, 2] < xyz_extent[2][1])]


        hist = self.hist2d(em_sub, xyz_extent[self.plot_axis[0]], xyz_extent[self.plot_axis[1]])
        
        if self.rel_clip is not None:
            hist = np.clip(hist, 0., hist.max()*self.rel_clip)
        if self.abs_clip is not None:
            hist = np.clip(hist, 0., self.abs_clip)  
            
        if self.sigma_blur is not None:
            hist = gaussian_filter(hist, sigma=[self.sigma_blur / self.px_size, self.sigma_blur / self.px_size])
            
        hist = np.clip(hist, 0, hist.max()/self.contrast)
            
        return torch.from_numpy(hist)

    def hist2d(self, em, x_hist_ext, y_hist_ext):

        xy = em.xyz_nm[:, self.plot_axis].numpy()
        
        hist_bins_x = np.arange(x_hist_ext[0], x_hist_ext[1] + self.px_size, self.px_size)
        hist_bins_y = np.arange(y_hist_ext[0], y_hist_ext[1] + self.px_size, self.px_size)

        hist, _, _ = np.histogram2d(xy[:, 0], xy[:, 1], bins=(hist_bins_x, hist_bins_y))

        return hist


class Renderer3D(Renderer):
    """
    3D Renderer with constant gaussian.
    
    Args:
        px_size: pixel size of the output image in nm
        sigma_blur: sigma of the gaussian blur applied in nm
        plot_axis: determines which dimensions get plotted. 0,1,2 = x,y,z. (0,1,2) is x over y, colored by z.
        xextent: extent in x in nm
        yextent: extent in y in nm
        zextent: extent in z in nm. 
        abs_clip: absolute clipping value of the histogram in counts
        rel_clip: clipping value relative to the maximum count. i.e. rel_clip = 0.8 clips at 0.8*hist.max()
        contrast: scaling factor to increase contrast
    """

    def __init__(self, px_size, sigma_blur, plot_axis=(0, 1, 2), xextent=None, yextent=None, zextent=None,
                 abs_clip=None, rel_clip=None, contrast=1):
        super().__init__(plot_axis=plot_axis, xextent=xextent, yextent=yextent, zextent=zextent, px_size=px_size, abs_clip=abs_clip, rel_clip=rel_clip, contrast=contrast)

        self.sigma_blur = sigma_blur
        self.zextent = zextent
        
        # get jet colormap
        lin_hue = np.linspace(0,1,256)
        cmap = plt.get_cmap('jet', lut=256);
        cmap = cmap(lin_hue)
        cmap_hsv = rgb_to_hsv(cmap[:,:3])
        jet_hue = cmap_hsv[:,0]
        _,b = np.unique(jet_hue, return_index=True)
        jet_hue = [jet_hue[index] for index in sorted(b)]
        self.jet_hue = np.interp(np.linspace(0,len(jet_hue),256), np.arange(len(jet_hue)), jet_hue)
        
    def render(self, em: emitter.EmitterSet, ax=None):

        hist = self.forward(em).numpy()

        if ax is None:
            ax = plt.gca()
            
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size=0.25, pad=-0.25)
        colb = mpl.colorbar.ColorbarBase(cax, cmap=plt.get_cmap('jet'), values=np.linspace(0, 1., 101),
                                         norm=mpl.colors.Normalize(0., 1.))
        colb.outline.set_visible(False)

        cax.text(0.12, 0.04, f'{self.zextent[0]} nm', rotation=90, color='white', fontsize=15, transform=cax.transAxes)
        cax.text(0.12, 0.88, f'{self.zextent[1]} nm', rotation=90, color='white', fontsize=15, transform=cax.transAxes)
        cax.axis('off')

        ax.imshow(np.transpose(hist, [1, 0, 2]))
        return ax

    def forward(self, em: emitter.EmitterSet) -> torch.Tensor:

        xyz_extent = self.get_extent(em)
        em_sub = em[(em.xyz_nm[:, 0] > xyz_extent[0][0]) * (em.xyz_nm[:, 0] < xyz_extent[0][1]) * 
                    (em.xyz_nm[:, 1] > xyz_extent[1][0]) * (em.xyz_nm[:, 1] < xyz_extent[1][1]) *
                    (em.xyz_nm[:, 2] > xyz_extent[2][0]) * (em.xyz_nm[:, 2] < xyz_extent[2][1])]

        int_hist, col_hist = self.hist2d(em_sub, xyz_extent[self.plot_axis[0]], xyz_extent[self.plot_axis[1]], xyz_extent[self.plot_axis[2]])
                
        with np.errstate(divide='ignore', invalid='ignore'):
            z_avg = col_hist / int_hist
            
        if self.rel_clip is not None:
            int_hist = np.clip(int_hist*self.contrast, 0., int_hist.max()*self.rel_clip)
            val = int_hist / int_hist.max()
        elif self.abs_clip is not None:
            int_hist = np.clip(int_hist, 0., self.abs_clip) 
            val = int_hist / self.abs_clip
        else:
            val = int_hist / int_hist.max()           
            
        val *= self.contrast
            
        z_avg[np.isnan(z_avg)] = 0
        sat = np.ones(int_hist.shape)
        hue = np.interp(z_avg,np.linspace(0,1,256),self.jet_hue)
        
        HSV = np.concatenate((hue[:, :, None], sat[:, :, None], val[:, :, None]), -1)
        RGB = hsv_to_rgb(HSV)

        if self.sigma_blur:
            RGB = np.array([gaussian_filter(RGB[:, :, i], sigma=[self.sigma_blur / self.px_size,
                                                                 self.sigma_blur / self.px_size]) for i in
                            range(3)]).transpose(1, 2, 0)
    
        RGB = np.clip(RGB, 0, 1)
        return torch.from_numpy(RGB)

    def hist2d(self, em, x_hist_ext, y_hist_ext, z_range):

        xyz = em.xyz_nm[:, self.plot_axis].numpy()
        
        hist_bins_x = np.arange(x_hist_ext[0], x_hist_ext[1] + self.px_size, self.px_size)
        hist_bins_y = np.arange(y_hist_ext[0], y_hist_ext[1] + self.px_size, self.px_size)

        int_hist, _, _ = np.histogram2d(xyz[:, 0], xyz[:, 1], bins=(hist_bins_x, hist_bins_y))

        z_pos = np.clip(xyz[:, 2], z_range[0], z_range[1])
        z_weight = ((z_pos - z_pos.min()) / (z_pos.max() - z_pos.min()))

        col_hist, _, _ = np.histogram2d(xyz[:, 0], xyz[:, 1], bins=(hist_bins_x, hist_bins_y), weights=z_weight)

        return int_hist, col_hist
    
class Renderer2D_auto_sig(Renderer2D):
    
    def __init__(self, px_size, batch_size=1000, filt_size=10, plot_axis = (0,1,2), xextent=None, yextent=None, zextent=None, abs_clip=None, rel_clip=None, contrast=1, device='cpu'):
        super().__init__(px_size=px_size, sigma_blur=None, plot_axis=plot_axis, xextent=xextent, yextent=yextent, zextent=zextent, abs_clip=abs_clip, rel_clip=rel_clip, contrast=contrast)

        self.sigma_scale = sigma_scale
        self.bs = batch_size
        self.fs = filt_size
        self.device = device

    def calc_gaussians(self, xy_mu, xy_sig, mesh):

        xy_mu =  xy_mu[:,:2] % self.px_size / self.px_size
        xy_sig = xy_sig[:,:2] / self.px_size
        
        dist = torch.distributions.Normal(xy_mu, xy_sig)
        W = torch.exp(dist.log_prob(mesh[:,:,None]).sum(-1)).permute(2,0,1)
        
        return (W/torch.clamp_min(W.sum(-1).sum(-1),1.)[:,None,None])    
    
    @script
    def place_gaussians(int_hist, inds, W, fs):
        for i in range(len(W)):
            int_hist[inds[i,1]:inds[i,1]+fs, inds[i,0]:inds[i,0]+fs] += W[i]      
        return int_hist
    
    def hist2d(self, em, x_hist_ext, y_hist_ext):

        ym, xm = torch.meshgrid(torch.linspace(-(self.fs//2),self.fs//2,self.fs, device=self.device), 
                                torch.linspace(-(self.fs//2),self.fs//2,self.fs, device=self.device))

        mesh = torch.cat([(xm)[...,None],(ym)[...,None]],-1)
                
        xy_mus = em.xyz_nm[:,self.plot_axis[:2]].to(self.device)
        xy_sigs = em.xyz_sig_nm[:,self.plot_axis[:2]].to(self.device)
        
        w = int((x_hist_ext[1]-x_hist_ext[0])//self.px_size+1)
        h = int((y_hist_ext[1]-y_hist_ext[0])//self.px_size+1)      
        
        int_hist = torch.zeros([h+self.fs,w+self.fs], device=self.device, dtype=torch.float)
        s_inds = ((xy_mus - torch.Tensor([x_hist_ext[0],y_hist_ext[0]]).to(self.device)) // self.px_size).type(torch.LongTensor)
        
        for i in tqdm(range(len(xy_mus)//self.bs)):

            sl = np.s_[i*self.bs:(i+1)*self.bs]
            sub_inds =  s_inds[sl]
            W = self.calc_gaussians(xy_mus[sl], xy_sigs[sl], mesh)
            int_hist = self.place_gaussians(int_hist, sub_inds, W, torch.tensor(self.fs))

        int_hist = int_hist[self.fs//2:-(self.fs//2+1),self.fs//2:-(self.fs//2+1)]
        
        return int_hist.T.numpy()

class Renderer3D_auto_sig(Renderer3D):
    
    def __init__(self, px_size, batch_size=1000, filt_size=10, plot_axis = (0,1,2), xextent=None, yextent=None, zextent=None, abs_clip=None, rel_clip=None, contrast=1, device='cpu'):
        super().__init__(px_size=px_size, sigma_blur=None, plot_axis=plot_axis, xextent=xextent, yextent=yextent, zextent=zextent, abs_clip=abs_clip, rel_clip=rel_clip, contrast=contrast)

        self.sigma_scale = sigma_scale
        self.bs = batch_size
        self.fs = filt_size
        self.device = device

    def calc_gaussians(self, xy_mu, xy_sig, mesh):

        xy_mu =  xy_mu[:,:2] % self.px_size / self.px_size
        xy_sig = xy_sig[:,:2] / self.px_size
        
        dist = torch.distributions.Normal(xy_mu, xy_sig)
        W = torch.exp(dist.log_prob(mesh[:,:,None]).sum(-1)).permute(2,0,1)
        
        return (W/torch.clamp_min(W.sum(-1).sum(-1),1.)[:,None,None])    
    
    @script
    def place_gaussians(comb_hist, inds, weights, W, fs):
        for i in range(len(W)):
            comb_hist[inds[i,1]:inds[i,1]+fs, inds[i,0]:inds[i,0]+fs] += torch.stack([W[i],W[i]*weights[i]],-1)      
        return comb_hist
    
    def hist2d(self, em, x_hist_ext, y_hist_ext, z_range):

        ym, xm = torch.meshgrid(torch.linspace(-(self.fs//2),self.fs//2,self.fs, device=self.device), 
                                torch.linspace(-(self.fs//2),self.fs//2,self.fs, device=self.device))

        mesh = torch.cat([(xm)[...,None],(ym)[...,None]],-1)
                
        xy_mus = em.xyz_nm[:,self.plot_axis[:2]].to(self.device)
        xy_sigs = em.xyz_sig_nm[:,self.plot_axis[:2]].to(self.device)
        
        z_pos = torch.clip(em.xyz_nm[:, self.plot_axis[2]], z_range[0], z_range[1])
        z_weight = ((z_pos - z_pos.min()) / (z_pos.max() - z_pos.min())).to(self.device)
        
        w = int((x_hist_ext[1]-x_hist_ext[0])//self.px_size+1)
        h = int((y_hist_ext[1]-y_hist_ext[0])//self.px_size+1)      
        
        comb_hist = torch.zeros([h+self.fs,w+self.fs,2], device=self.device, dtype=torch.float)
        
        s_inds = ((xy_mus - torch.Tensor([x_hist_ext[0],y_hist_ext[0]]).to(self.device)) // self.px_size).type(torch.LongTensor)
        
        for i in tqdm(range(len(xy_mus)//self.bs)):

            sl = np.s_[i*self.bs:(i+1)*self.bs]
            sub_inds =  s_inds[sl]
            z_ws = z_weight[sl]
            W = self.calc_gaussians(xy_mus[sl], xy_sigs[sl], mesh)
            
            comb_hist = self.place_gaussians(comb_hist, sub_inds, z_ws, W, torch.tensor(self.fs))

        comb_hist = comb_hist[self.fs//2:-(self.fs//2+1),self.fs//2:-(self.fs//2+1)]
        int_hist = comb_hist[:,:,0]
        col_hist = comb_hist[:,:,1]
        
        return int_hist.T.numpy(), col_hist.T.numpy()