import numpy as np
from scipy.ndimage.measurements import label
from skimage.feature import peak_local_max
from sklearn.cluster import DBSCAN
import torch

from deepsmlm.generic.coordinate_trafo import UpsamplingTransformation as ScaleTrafo
from deepsmlm.generic.coordinate_trafo import A2BTransform
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

        self.transformation = ScaleTrafo(self.extent, self.upsampling_factor)

    def forward(self, img):
        cord = np.ascontiguousarray(peak_local_max(img.detach().numpy(),
                                                         min_distance=self.min_distance,
                                                         threshold_abs=self.threshold,
                                                         exclude_border=False))

        cord = torch.from_numpy(cord)
        return self.transformation.up2coord(cord)


class CoordScan:
    """Cluster to coordinate midpoint post processor"""
    def __init__(self, cluster_dims, eps=0.5, phot_threshold=0.8, clusterer=None):

        self.cluster_dims = cluster_dims
        self.eps = eps
        self.phot_tr = phot_threshold

        if clusterer is None:
            self.clusterer = DBSCAN(eps=eps, min_samples=phot_threshold)

    def forward(self, xyz, phot):
        """
        Forward a batch of list of coordinates through the clustering algorithm.

        :param xyz: batchised coordinates (Batch x N x D)
        :param phot: batchised photons (Batch X N)
        :return: list of tensors of clusters, and list of tensor of photons
        """
        assert xyz.dim() == 3
        batch_size = xyz.shape[0]

        xyz_out = [None] * batch_size
        phot_out = [None] * batch_size

        """Loop over the batch"""
        for i in range(batch_size):
            xyz_ = xyz[i, :, :].numpy()
            phot_ = phot[i, :].numpy()

            if self.cluster_dims == 2:
                db = self.clusterer.fit(xyz_[:, :2], phot_)
            else:
                core_samples, clus_ix = self.clusterer.fit(xyz_, phot_)

            core_samples = db.core_sample_indices_
            clus_ix = db.labels_

            core_samples = torch.from_numpy(core_samples)
            clus_ix = torch.from_numpy(clus_ix)
            num_cluster = clus_ix.max() + 1  # because -1 means not in cluster, and then from 0 - max_ix

            xyz_batch_cluster = torch.zeros((num_cluster, xyz_.shape[1]))
            phot_batch_cluster = torch.zeros(num_cluster)

            """Loop over the clusters"""
            for j in range(num_cluster):
                in_clus = clus_ix == j

                xyz_clus = xyz_[in_clus, :]
                phot_clus = phot_[in_clus]

                """Calculate weighted average. Maybe replace by (weighted) median?"""
                clus_mean = np.average(xyz_clus, axis=0, weights=phot_clus)
                xyz_batch_cluster[j, :] = torch.from_numpy(clus_mean)
                photons = phot_clus.sum()
                phot_batch_cluster[j] = photons

            xyz_out[i] = xyz_batch_cluster
            phot_out[i] = phot_batch_cluster

        return xyz_out, phot_out


class ConnectedComponents:
    def __init__(self, photon_threshold, extent, clusterer=label, single_value_threshold=0):
        self.phot_thres = photon_threshold
        self.single_val_threshold = single_value_threshold
        self.extent = extent
        self.dim = 2 if (extent[2] is None) else 3
        self.clusterer = clusterer
        self.matrix_extent = None

        self.kernel = np.ones((3, 3))

    def forward(self, x):
        """
        Forward a single frame (currently not batchised).

        :param x: 2D frame 1 channel
        :return: cluster centres
        """

        def cluster_average(coords, photons, cluster_ix_p_coord):
            num_clusters = cluster_ix_p_coord.max()
            print("Number of cluster: {}".format(num_clusters))
            pos_av = np.zeros((num_clusters, coords.shape[1]))
            phot_sum = np.zeros((num_clusters,))
            for i in range(num_clusters):
                # ix in current cluster
                ix = (cluster_ix_p_coord == i + 1)

                if ix.sum() == 0:
                    print("No coordinates found. Skipping. {}.".format(i))
                    continue

                """Calculate position average."""
                pos_av[i, :] = np.average(coords[ix, :], axis=0, weights=photons[ix])
                phot_sum[i] = np.sum(photons[ix], axis=0)
            return pos_av, phot_sum

        """Threshold single px values"""
        x_ = x.numpy()
        self.matrix_extent = ((-0.5, x_.shape[0] - 0.5), (-0.5, x_.shape[1] - 0.5))
        x_[x_ < self.single_val_threshold] = 0

        cluster_frame, num_clusters = label(x_, self.kernel)
        # get the coordinates of the cluster members
        clus_bool = cluster_frame >= 1
        clus_ix = cluster_frame[clus_bool]
        phot_in_clus = x_[clus_bool]
        clus_mat_coord = np.asarray(np.asarray(clus_bool).nonzero()).transpose()

        pos_clus, phot_clus = cluster_average(clus_mat_coord, phot_in_clus, clus_ix)
        pos_clus, phot_clus = torch.from_numpy(pos_clus), torch.from_numpy(phot_clus)

        # Transform coordinates
        pos_clus = A2BTransform(a_extent=self.matrix_extent, b_extent=self.extent).a2b(pos_clus)

        """Filter by photon threshold"""
        ix_above_thres = phot_clus > self.phot_thres
        return pos_clus[ix_above_thres, :], phot_clus[ix_above_thres]



if __name__ == '__main__':
    from sklearn.datasets.samples_generator import make_blobs

    # x = torch.tensor([[25., 25., 0], [0., 0., 5.], [0., 0., 7]])
    # xyz = torch.tensor([[0.0, 0.0], [0.1, 0.05], [5.2, 5], [5.3, 5.1]])
    # phot = torch.tensor([0.4, 0.4, 0.4, 0.2])
    # cn = ConnectedComponents(mode='coords',
    #                          distance_threshold=0.0015,
    #                          photon_threshold=0.6,
    #                          extent=((-0.5, 10), (-0.5, 10), None))
    # xyz_clus, phot_clus = cn.forward(xyz, phot)
    # print(xyz_clus, phot_clus)

    frame = torch.zeros((50, 50))
    frame[5, 5] = 0.3
    frame[5, 6] = 0.5
    frame[4, 5] = 0.2

    cn = ConnectedComponents(photon_threshold=0.6,
                             extent=((-0.5, 24.5), (-0.5, 24.5), None))
    xyz_clus, phot_clus = cn.forward(frame)
    print(xyz_clus, phot_clus)

    # centers = [[1, 1], [-1, -1], [1, -1]]
    # X, labels_true = make_blobs(n_samples=750, centers=centers, cluster_std=0.4,
    #                             random_state=0)
    # X = torch.from_numpy(X).unsqueeze(0)
    # photons = torch.ones_like(X[:, :, 0])
    #
    # clusterer = CoordScan(2, 0.5, 5)
    # clus_means, clus_photons = clusterer.forward(X, photons)

    print("Success.")
