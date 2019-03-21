import numpy as np
from scipy.ndimage.measurements import label
from skimage.feature import peak_local_max
import torch

from deepsmlm.generic.coordinate_trafo import UpsamplingTransformation as trafo
from deepsmlm.generic.psf_kernel import DeltaPSF


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


class ConnectedComponents:
    def __init__(self, mode, threshold, extent):
        self.mode = mode
        self.threshold = threshold
        self.extent = extent
        self.dim = 2 if (extent[2] is None) else 3

        """Bin according to specification."""

        shape_x = (self.extent[0][1] - self.extent[0][0]) / self.threshold
        shape_y = (self.extent[1][1] - self.extent[1][0]) / self.threshold
        if self.dim == 2:
            self.corner = (self.extent[0][0], self.extent[1][0])
            image_shape = (shape_x, shape_y)
        else:
            self.corner = (self.extent[0][0], self.extent[1][0], self.extent[2][0])
            shape_z = (self.extent[2][1] - self.extent[2][0]) / self.threshold
            image_shape = (shape_x, shape_y, shape_z)

        self.psf = DeltaPSF(self.extent[0], self.extent[1], self.extent[2], image_shape)

    def forward(self, x, phot=None):
        def round2base(v, base, mode=np.floor):
            return base * mode(v / base)

        def coord_2_cluster_ix(coords, corner, bin_width, clusters):
            # floor ccoords to bin_edges
            assert bin_width[0] == bin_width[1]
            bin_width = bin_width[0]
            coords_ = coords - np.array(corner)
            coords_floored = round2base(coords_, bin_width, np.floor)

            # find bin ix
            bin_ix = (coords_floored / bin_width).astype(int)

            # return cluster_ix per coord
            return clusters[bin_ix[:, 0], bin_ix[:, 1]]

        def cluster_average(coords, photons, cluster_ix_p_coord):
            num_clusters = cluster_ix_p_coord.max()
            pos_av = np.zeros((num_clusters, coords.shape[1]))
            phot_sum = np.zeros((num_clusters,))
            for i in range(num_clusters):
                # ix in current cluster
                ix = (cluster_ix_p_coord == i + 1)

                if ix.sum() == 0:
                    print("Fail. {}".format(i))
                    return None

                # calculate pos_av
                pos_av[i, :] = np.average(coords[ix, :], axis=0, weights=photons[ix])
                phot_sum[i] = np.sum(photons[ix], axis=0)
            return pos_av, phot_sum

        x = x.numpy()

        if self.mode == 'coords':
            assert phot is not None
            phot = phot.numpy()

            frame = self.psf.forward(x, phot).squeeze()
            kernel = np.ones((3, 3))
            cluster_frame, _ = label(frame, kernel)

            clusix = coord_2_cluster_ix(x, self.corner, self.threshold, cluster_frame)
            pos_clus, phot_clus = cluster_average(x, phot, clusix)

            return pos_clus, phot_clus


        elif self.mode == 'frame':
            pass
        else
            raise ValueError("Wrong switch for mode of connected components.")