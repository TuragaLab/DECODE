import numpy as np
from scipy.ndimage.measurements import label
# from skimage.measure import label
from skimage.feature import peak_local_max
from sklearn.cluster import DBSCAN
import torch

from deepsmlm.generic.coordinate_trafo import UpsamplingTransformation as ScaleTrafo
from deepsmlm.generic.coordinate_trafo import A2BTransform
from deepsmlm.generic.psf_kernel import DeltaPSF, OffsetPSF
from deepsmlm.generic.emitter import EmitterSet


class PeakFinder:
    """
    Class to find a local peak of the network output.
    This is similiar to non maximum suppresion.
    """

    def __init__(self, threshold, min_distance, extent, upsampling_factor):
        """
        Documentation from http://scikit-image.org/docs/dev/api/skimage.feature.html#skimage.feature.peak_local_max if
        parameters applicable to peak_local_max

        :param threshold: Minimum intensity of peaks. By default, the absolute threshold is the minimum intensity of the image.
        :param min_distance: Minimum number of pixels separating peaks in a region of 2 * min_distance + 1 (i.e. peaks
         are separated by at least min_distance). To find the maximum number of peaks, use min_distance=1.
        :param extent: extent of the input image
        :param upsampling_factor: factor by which input image is upsampled
        """
        self.threshold = threshold
        self.min_distance = min_distance
        self.extent = extent
        self.upsampling_factor = upsampling_factor
        self.transformation = ScaleTrafo(self.extent, self.upsampling_factor)

    def forward(self, img):
        """
        Forward img to find the peaks (a way of declustering).
        :param img: batchised image --> N x C x H x W
        :return: emitterset
        """
        if img.dim() != 4:
            raise ValueError("Wrong dimension of input image. Must be N x C=1 x H x W.")

        n_batch = img.shape[0]
        coord_batch = []
        img_ = img.detach().numpy()
        for i in range(n_batch):
            cord = np.ascontiguousarray(peak_local_max(img_[i, 0, :, :],
                                                       min_distance=self.min_distance,
                                                       threshold_abs=self.threshold,
                                                       exclude_border=False))

            cord = torch.from_numpy(cord)

            # Transform cord based on image to cord based on extent
            cord = self.transformation.up2coord(cord)
            coord_batch.append(EmitterSet(cord,
                                        (torch.ones(cord.shape[0]) * (-1)),
                                        frame_ix=torch.zeros(cord.shape[0])))

        return coord_batch


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
    def __init__(self, prob_th, svalue_th=0, connectivity=2):
        self.prob_th = prob_th
        self.svalue_th = svalue_th
        self.clusterer = label

        if connectivity == 2:
            self.kernel = np.ones((3, 3))
        elif connectivity == 1:
            self.kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])

    def compute_cix(self, p_map):
        """Computes cluster ix, based on a prob-map.

        :param p_map: either N x H x W or H x W
        :return: cluster indices (NHW or HW)
        """
        if p_map.dim() == 2:
            out_hw = True  # output in hw format, i.e. without N
            p = p_map.clone().unsqueeze(0)
        else:
            out_hw = False  # output in NHW format
            p = p_map.clone()

        cluster_ix = torch.zeros_like(p)

        for i in range(p.size(0)):
            c_ix, _ = label(p[i].numpy(), self.kernel)
            cluster_ix[i] = torch.from_numpy(c_ix)

        if out_hw:
            return cluster_ix.squeeze(0)
        else:
            return cluster_ix


class CC5ChModel(ConnectedComponents):
    """Connected components on 5 channel model."""
    def __init__(self, prob_th, svalue_th=0, connectivity=2):
        super().__init__(prob_th, svalue_th, connectivity)

    @staticmethod
    def average_features(features, cluster_ix, weight):
        """
        Averages the features per cluster weighted by the probability.

        :param features: tensor (N)CHW
        :param cluster_ix: (N)HW
        :param weight: (N)HW
        :return: list of tensors of size number of clusters x features
        """

        """Add batch dimension if not already present."""
        if features.dim() == 3:
            red2hw = True  # squeeze batch dim out for return
            features = features.unsqueeze(0)
            cluster_ix = cluster_ix.unsqueeze(0)
            weight = weight.unsqueeze(0)
        else:
            red2hw = False

        batch_size = features.size(0)

        """Flatten features, weights and cluster_ix in image space"""
        feat_flat = features.view(batch_size, features.size(1), -1)
        clusix_flat = cluster_ix.view(batch_size, -1)
        w_flat = weight.view(batch_size, -1)

        feat_av = []  # list of feature average tensors
        p = []  # list of cumulative probabilites

        """Loop over the batches"""
        for i in range(batch_size):
            ccix = clusix_flat[i]  # current cluster indices in batch
            num_clusters = int(ccix.max().item())

            feat_i = torch.zeros((num_clusters, feat_flat.size(1)))
            p_i = torch.zeros(num_clusters)

            for j in range(num_clusters):
                # ix in current cluster
                ix = (ccix == j + 1)

                if ix.sum() == 0:
                    continue

                feat_i[j, :] = torch.from_numpy(
                    np.average(feat_flat[i, :, ix].numpy(), axis=1, weights=w_flat[i, ix].numpy()))
                p_i[j] = feat_flat[i, 0, ix].sum()

            feat_av.append(feat_i)
            p.append(p_i)

        if red2hw:
            return feat_av[0], p[0]
        else:
            return feat_av, p

    def forward(self, ch5_input):
        """
        Forward a batch of output of 5ch model.
        :param ch5_input: N x C=5 x H x W
        :return: emitterset
        """

        if ch5_input.dim() == 3:
            red_batch = True  # squeeze out batch dimension in the end
            ch5_input = ch5_input.unsqueeze(0)
        else:
            red_batch = False

        batch_size = ch5_input.size(0)

        """Compute connected components based on prob map."""
        p_map = ch5_input[:, 0]
        clusters = self.compute_cix(p_map)

        """Average the within cluster features"""
        feature_av, prob = self.average_features(ch5_input, clusters, p_map)  # returns list tensors of averaged feat.

        """Return list of emittersets"""
        emitter_sets = [None] * batch_size

        for i in range(batch_size):
            pi = prob[i]
            feat_i = feature_av[i]

            # get list of emitters:
            p_red = pi
            phot_red = feat_i[:, 1]
            xyz = torch.cat((
                feat_i[:, 2].unsqueeze(1),
                feat_i[:, 3].unsqueeze(1),
                feat_i[:, 4].unsqueeze(1)), 1)

            em = EmitterSet(xyz, phot_red, frame_ix=(i * torch.ones_like(phot_red)))

            emitter_sets[i] = em
        return emitter_sets


class Offset2Coordinate:
    """Postprocesses the offset model to return a list of emitters."""
    def __init__(self, xextent, yextent, img_shape):

        off_psf = OffsetPSF(xextent=xextent,
                            yextent=yextent,
                            img_shape=img_shape)

        xv, yv = torch.meshgrid([off_psf.bin_ctr_x, off_psf.bin_ctr_y])
        self.x_mesh = xv.unsqueeze(0)
        self.y_mesh = yv.unsqueeze(0)

    def _convert_xy_offset(self, x_offset, y_offset):
        batch_size = x_offset.size(0)
        x_coord = self.x_mesh.repeat(batch_size, 1, 1) + x_offset
        y_coord = self.y_mesh.repeat(batch_size, 1, 1) + y_offset
        return x_coord, y_coord

    def forward(self, output):
        """
        Forwards a batch of 5ch offset model and convert the offsets to coordinates
        :param output:
        :return:
        """

        """Convert to batch if not already is one"""
        if output.dim() == 3:
            squeeze_batch_dim = True
            output = output.unsqueeze(0)
        else:
            squeeze_batch_dim = False

        """Convert the channel values to coordinates"""
        x_coord, y_coord = self._convert_xy_offset(output[:, 2], output[:, 3])

        output_converted = output.clone()
        output_converted[:, 2] = x_coord
        output_converted[:, 3] = y_coord

        if squeeze_batch_dim:
            return output_converted.squeeze(0)
        else:
            return output_converted


class ConnectedComponentsOld:
    def __init__(self, photon_threshold, extent, clusterer=label, single_value_threshold=0, connectivity=2):
        """

        :param photon_threshold: minimum total value of summmed output
        :param extent:
        :param clusterer:
        :param single_value_threshold:
        :param connectivity:
        """
        self.phot_thres = photon_threshold
        self.single_val_threshold = single_value_threshold
        self.extent = extent
        self.dim = 2 if (extent[2] is None) else 3
        self.clusterer = clusterer
        self.matrix_extent = None
        self.connectivity = connectivity

        if connectivity == 2:
            self.kernel = np.ones((3, 3))
        elif connectivity == 1:
            self.kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])

    @staticmethod
    def _generate_coordinate_mesh(mat_shape):
        """
        Helper method to generate coordinates when the mask for connected labels is at the same time the positions.
        :return:
        """
        """Generate a meshgrid where the first component is x, 
        the second is y. concat them as channels to feed it into the averager.
        """
        xv, yv = torch.meshgrid([torch.arange(0, mat_shape[0]), torch.arange(0, mat_shape[1])])
        return torch.cat((xv.unsqueeze(0), yv.unsqueeze(0)), 0)

    @staticmethod
    def channeled_average(cmpt_ix, ch_input, weight):

        # flatten features, weights and mask (C x H x W to C x HW)
        mask_flat = cmpt_ix.view(-1)
        w_flat = weight.view(-1)
        ch_flat = ch_input.view(ch_input.size(0), -1)

        num_clusters = cmpt_ix.max()
        feat_av = torch.zeros((num_clusters, ch_input.size(0)))
        p_sum = torch.zeros(num_clusters)

        for i in range(num_clusters):
            # ix in current cluster
            ix = (cmpt_ix == i + 1)

            if ix.sum() == 0:
                continue

            # calculate the average over all features
            feat_av[i, :] = torch.from_numpy(np.average(ch_input[:, ix].numpy(), axis=1, weights=w_flat.numpy()))
            p_sum[i] = ch_input[0, ix].sum()

        return feat_av, p_sum

    def forward(self, x):
        """
        Forward a batch of frames through connected components. Must only contain one channel.

        :param x: 2D frame, or 4D batch of 1 channel frames.
        :return: (instance of emitterset)
        """

        # loop over all batch elements
        if not(x.dim() == 3 or x.dim() == 4):
            raise ValueError("Input must be C x H x W or N x C x H x W.")
        elif x.dim() == 3:
            x_ = x.unsquueze(0)
        else:
            x_ = x

        clusters = []

        for i in range(x_.shape[0]):

            """Threshold single px values"""
            mask = x_[i, 0, :, :]
            x_ = x_[i, :, :, :]
            prob = x_[0]

            # self.matrix_extent = ((-0.5, mask.shape[0] - 0.5), (-0.5, mask.shape[1] - 0.5))
            mask[x_ < self.single_val_threshold] = 0

            cmpt_ix, num_clusters = label(mask.numpy(), self.kernel)
            cmpt_ix = torch.from_numpy(cmpt_ix)

            feat_av, p_sum = self.channeled_average(cmpt_ix, x_, prob)

            """Filter by prob threshold"""
            ix_above_thres = p_sum > self.phot_thres
            # p_sum = p_sum[ix_above_thres]

            """Calc xyz"""
            feat_av = feat_av[ix_above_thres, :]
            xyz = torch.cat((feat_av[:, [2]], feat_av[:, [3]], feat_av[:, [4]]), 1)
            phot = feat_av[:, 1]

            clusters.append(EmitterSet(xyz=xyz,
                                       phot=phot,
                                       frame_ix=(torch.ones_like(phot) * (-1))))

        return clusters


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
    frame[0, 0] = 0.3
    frame[1, 1] = 0.5
    frame[2, 2] = 0.2

    cn = ConnectedComponents(photon_threshold=0.6,
                             extent=((-0.5, 24.5), (-0.5, 24.5), None), connectivity=1)
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
