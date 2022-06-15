import decode
import decode.utils

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import glob

import math
import warnings
from abc import ABC, abstractmethod
from typing import Tuple, Optional, Union

import numpy as np
import spline  # cubic spline implementation
import torch

import decode.generic.utils

class PSFVolInterpolation(ABC):
    def __init__(
        self,
        psf_cube: torch.tensor,
        slice_mode: bool,
        device: str = "cuda:0",
    ):    
        """
        Abstract class that represents functions for performing subpixel interpolations on PSFs 
        which are represented as volumes.

        Args:
            slide_mode: whether to interpolate whole volumes, or select slices before interpolation
        """

        self.psf_cube = psf_cube.to(device)
        assert self.psf_cube.ndim == 4, 'Has to be 4 dimensional (color, z, y, x)'
        
        self.int_vol_shape_xyz = list(psf_cube.shape[1:][::-1])
        
        self.n_colors = psf_cube.shape[0]
        self.slice_mode = slice_mode
        self._device = device
        
        self.mesh_vec = [torch.linspace(-1+1/sz, 1-1/sz, int(sz)).to(self._device) for sz in self.int_vol_shape_xyz]     
        
        if self.slice_mode:
            self.mesh_vec[2] = torch.zeros(1).to(self._device)
            self.int_vol_shape_xyz[2] = 3
            
        self.xyz_grid = torch.meshgrid(*self.mesh_vec)
        self.xy_grid = torch.meshgrid(*self.mesh_vec[:2])
        
    def forward(
        self,
        xyz_shift: torch.Tensor,
        z_inds: Optional[torch.Tensor] = None, 
        col_inds: Optional[torch.Tensor] = None):
        
        """
        Performs subpixel shifts for the given coordinates using some interpolation method
        If in slice mode, also indexes the correct slices. 

        Args:
            xyz: coordinates (in pixels)
            z_inds: indices of z-slices (same length as xyz) (only if slice_mode = True)
            col_inds: indices of colors (only if n_colors > 1)

        Returns:
            psf_shifted: psf_volumes or slices shifted by xyz

        """
        
        n_rois = xyz_shift.shape[0]  # number of rois / emitters / fluorophores
        
        if self.slice_mode:
            assert z_inds is not None, "No z indices for slice mode data"
            assert z_inds.min() >= 1 and z_inds.max() <= self.psf_cube.shape[1] - 1, "z indices out of valid range [1, psf_stack_size - 1]"

            # We select a 3 pixel wide volume around the indexed z slice for interpolation
            forward_vol = torch.cat([self.psf_cube[None,:,[z-1 for z in z_inds]], 
                                     self.psf_cube[None,:,z_inds], 
                                     self.psf_cube[None,:,[z+1 for z in z_inds]]], dim=0).transpose(0,2).to(self._device)

        else:
            forward_vol = self.psf_cube
            forward_vol = forward_vol[None].expand(n_rois, -1, -1, -1, -1)

        if self.n_colors > 1:
            assert col_inds is not None, "No color indices for multi-color PSF"
            forward_vol = forward_vol[torch.arange(len(col_inds)),col_inds]
            forward_vol = forward_vol[:, None]
            
        return self.forward_vol(forward_vol, xyz_shift)
    
    def forward_vol(self, vol, xyz_shifts):
        return vol
                
    def shift_mesh_grid_3d(self, xyz_shifts):
        """ Creates a grid for interpolation in x,y,z and shifts it."""
        xyz_offsets = [2 * xyz_shifts[:, i] / self.int_vol_shape_xyz[i] for i in range(3)]        
        m_grid = torch.stack([self.xyz_grid[i][None] - xyz_offsets[i].to(self._device).view(-1, 1, 1, 1) for i in range(3)], -1)
        
        return m_grid
    
    def shift_mesh_grid_2d(self, xy_shifts):
        """ Creates a grid for interpolation in x,y and shifts it."""
        xy_offsets = [2 * xy_shifts[:, i] / self.int_vol_shape_xyz[i] for i in range(2)]
        m_grid = torch.stack([self.xy_grid[i][None] - xy_offsets[i].to(self._device).view(-1, 1, 1) for i in range(2)], -1)
        
        return m_grid
        
class TrilinearInterpolation(PSFVolInterpolation): 
    """
    Performs trilinear interpolation in xyz
    """
    
    def __init__(
        self,
        psf_cube: torch.tensor,
        slice_mode: bool,
        device: str = "cuda:0",
    ):     
        
        super().__init__(
            psf_cube=psf_cube, slice_mode=slice_mode, device=device
        )
        
    def forward_vol(self, 
               vol: torch.tensor,
               xyz_shifts: torch.tensor):
        
        m_grid = self.shift_mesh_grid_3d(xyz_shifts)
        
        psf_shifted = torch.nn.functional.grid_sample(vol, m_grid, align_corners = False, mode='bilinear')
        psf_shifted = psf_shifted.transpose(-3,-1)
        
        return psf_shifted  
        
class BicubicInterpolation(PSFVolInterpolation):
    """
    Performs bilinear interpolation in z and bicubic interpolation in xy afterwards
    """    
    def __init__(
        self,
        psf_cube: torch.tensor,
        slice_mode: bool,
        device: str = "cuda:0",
    ):     
        
        super().__init__(
            psf_cube=psf_cube, slice_mode=slice_mode, device=device
        )
        
    def forward_vol(self, 
               vol: torch.tensor,
               xyz_shifts: torch.tensor):

        z_shifts = xyz_shifts + 0
        z_shifts[:,:2] = 0.
        m_grid_z = self.shift_mesh_grid_3d(z_shifts)
        
        psf_z_shifted = torch.nn.functional.grid_sample(vol, m_grid_z, align_corners = False, mode='bilinear')
        psf_z_shifted = psf_z_shifted.transpose(1,-1)[...,0] # Swap z into channel dim, and drop z dim.         
        
        m_grid_xy = self.shift_mesh_grid_2d(xyz_shifts[:,:2])
        
        psf_shifted = torch.nn.functional.grid_sample(psf_z_shifted, m_grid_xy, align_corners = False, mode='bicubic')
        psf_shifted = psf_shifted.transpose(-2,-1)[:, None]

        return psf_shifted  