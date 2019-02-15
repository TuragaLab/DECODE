import numpy as np
from skimage.feature import peak_local_max
import torch

from deepsmlm.generic.coordinate_trafo import UpsamplingTransformation as trafo


class PeakFinder:
    """
    Class to find a local peak of the network output.
    This is similiar to non maximum suppresion.
    """

    def __init__(self, threshold, min_distance, extent, upsampling_factor):
        self.threshold = threshold
        self.min_distance = min_distance
        self.extent = extent
        self.upsampling_factor = upsampling_factor

        self.transformation = trafo(self.extent, self.upsampling_factor)

    def forward(self, img):
        cord = np.ascontiguousarray(peak_local_max(img.detach().numpy(),
                                                         min_distance=self.min_distance,
                                                         threshold_abs=self.threshold,
                                                         exclude_border=False))

        cord = torch.from_numpy(cord)
        return self.transformation.up2coord(cord)
