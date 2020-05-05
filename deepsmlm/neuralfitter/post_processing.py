import math
import warnings
from abc import ABC, abstractmethod  # abstract class

import numpy as np
import torch
from deprecated import deprecated
from joblib import Parallel, delayed
from sklearn.cluster import AgglomerativeClustering, DBSCAN

import deepsmlm.generic.utils.statistics as fan_stat
import deepsmlm.simulation.background
from deepsmlm.evaluation import match_emittersets
from deepsmlm.generic.emitter import EmitterSet, EmptyEmitterSet
from deepsmlm.neuralfitter.target_generator import UnifiedEmbeddingTarget


class PostProcessing(ABC):
    _return_types = ('batch-set', 'frame-set')

    def __init__(self, return_format: str):
        """

        Args:
            return_format (str): return format of forward function. Must be 'batch-set', 'frame-set'. If 'batch-set'
            one instance of EmitterSet will be returned per forward call, if 'frame-set' a tuple of EmitterSet one
            per frame will be returned
            sanity_check (bool): perform sanity check
        """

        super().__init__()
        self.return_format = return_format

    def sanity_check(self):
        """
        Sanity checks
        """
        if self.return_format not in self._return_types:
            raise ValueError("Not supported return type.")

    def skip_if(self, x):
        """
        Skip post-processing when a certain condition is met and implementation would fail, i.e. to many
        bright pixels in the detection channel. Default implementation returns False always.

        Args:
            x: network output

        Returns:
            bool: returns true when post-processing should be skipped
        """
        return False

    def _return_as_type(self, em, ix_low, ix_high):
        """
        Returns in the type specified in constructor

        Args:
            em (EmitterSet): emitters
            ix_low (int): lower frame_ix
            ix_high (int): upper frame_ix

        Returns:
            EmitterSet or list: Returns as EmitterSet or as list of EmitterSets

        """

        if self.return_format == 'batch-set':
            return em
        elif self.return_format == 'frame-set':
            return em.split_in_frames(ix_low=ix_low, ix_up=ix_high)
        else:
            raise ValueError

    @abstractmethod
    def forward(self, x):
        """
        Forward anything through the post-processing and return an EmitterSet

        Args:
            x:

        Returns:
            EmitterSet or list: Returns as EmitterSet or as list of EmitterSets

        """
        raise NotImplementedError


class NoPostProcessing(PostProcessing):
    """
    The 'No' Post-Processing. Just a helper.
    """

    def __init__(self, return_format='batch-set'):
        super().__init__(return_format=return_format)

    def forward(self, x: torch.Tensor):
        """

        Args:
            x (torch.Tensor): any input tensor where the first dim is the batch-dim.

        Returns:
            EmptyEmitterSet: An empty EmitterSet

        """

        em = EmptyEmitterSet()
        return self._return_as_type(em, ix_low=0, ix_high=x.size(0))


class ConsistencyPostprocessing(PostProcessing):
    """
    PostProcessing implementation that divides the output in hard and easy samples. Easy samples are predictions in
    which we have a single one hot pixel in the detection channel, hard samples are pixels in the detection channel
    where the adjacent pixels are also active (i.e. above a certain initial threshold).
    """
    _p_aggregations = ('sum', 'max', 'pbinom_cdf', 'pbinom_pdf')
    _xy_unit = 'nm'

    def __init__(self, *, raw_th, em_th, xy_unit: str, img_shape, ax_th=None, vol_th=None, lat_th=None,
                 p_aggregation='pbinom_cdf', px_size=None, match_dims=2, diag=0, num_workers=0,
                 return_format='batch-set', sanity_check=True):
        """

        Args:
            raw_th:
            em_th:
            xy_unit:
            img_shape:
            ax_th:
            vol_th:
            lat_th:
            p_aggregation:
            px_size:
            match_dims:
            diag:
            num_workers:
            return_format:
            sanity_check:
        """
        super().__init__(return_format=return_format)

        self.raw_th = raw_th
        self.em_th = em_th
        self.xy_unit = xy_unit
        self.px_size = px_size
        self.p_aggregation = p_aggregation
        self.match_dims = match_dims
        self.num_workers = num_workers

        self._filter = match_emittersets.GreedyHungarianMatching(match_dims=match_dims, dist_lat=lat_th,
                                                                 dist_ax=ax_th, dist_vol=vol_th).filter

        self._bg_calculator = deepsmlm.simulation.background.BgPerEmitterFromBgFrame(filter_size=13, xextent=(0., 1.),
                                                                                     yextent=(0., 1.),
                                                                                     img_shape=img_shape)

        self._neighbor_kernel = torch.tensor([[diag, 1, diag], [1, 1, 1], [diag, 1, diag]]).float().view(1, 1, 3, 3)

        self._clusterer = AgglomerativeClustering(n_clusters=None,
                                                  distance_threshold=lat_th if self.match_dims == 2 else vol_th,
                                                  affinity='precomputed',
                                                  linkage='single')

        if sanity_check:
            self.sanity_check()

    @staticmethod
    def parse(param):
        """
        Return an instance of this post-processing as specified by the parameters

        Args:
            param:

        Returns:
            ConsistencyPostProcessing

        """
        return ConsistencyPostprocessing(raw_th=param.PostProcessing.single_val_th, em_th=param.PostProcessing.total_th,
                                         xy_unit='px', px_size=param.Camera.px_size,
                                         img_shape=param.Simulation.img_size,
                                         ax_th=param.PostProcessing.ax_th, vol_th=param.PostProcessing.vol_th,
                                         lat_th=param.PostProcessing.lat_th, match_dims=param.PostProcessing.match_dims,
                                         return_format='batch-set')

    def sanity_check(self):
        """
        Performs some sanity checks. Part of the constructor; useful if you modify attributes later on and want to
        double check.

        """

        super().sanity_check()

        if self.p_aggregation not in self._p_aggregations:
            raise ValueError("Unsupported probability aggregation type.")

    def skip_if(self, x):  # ToDo: Implement this in forward
        if x.dim() != 4:
            raise ValueError("Unsupported dim.")

        if (x[:, 0] >= self.raw_th).sum() > 0.2 * x[:, 0].numel():
            return True
        else:
            return False

    def _cluster_batch(self, p, features):
        """
        Cluster a batch of frames
        
        Args:
            p (torch.Tensor): detections
            features (torch.Tensor): features

        Returns:

        """

        clusterer = self._clusterer
        p_aggregation = self.p_aggregation

        if p.size(1) > 1:
            raise ValueError("Not Supported shape for propbabilty.")
        p_out = torch.zeros_like(p).view(p.size(0), p.size(1), -1)
        feat_out = features.clone().view(features.size(0), features.size(1), -1)

        """Frame wise clustering."""
        for i in range(features.size(0)):
            ix = p[i, 0] > 0
            if (ix == 0.).all().item():
                continue
            alg_ix = (p[i].view(-1) > 0).nonzero().squeeze(1)

            p_frame = p[i, 0, ix].view(-1)
            f_frame = features[i, :, ix]
            # flatten samples and put them in the first dim
            f_frame = f_frame.reshape(f_frame.size(0), -1).permute(1, 0)

            filter_mask = self._filter(f_frame[:, 1:4], f_frame[:, 1:4])
            if self.match_dims == 2:
                dist_mat = torch.cdist(f_frame[:, 1:3], f_frame[:, 1:3])
            elif self.match_dims == 3:
                dist_mat = torch.cdist(f_frame[:, 1:4], f_frame[:, 1:4])
            else:
                raise ValueError

            dist_mat[~filter_mask] = 999999999999.  # those who shall not match shall be separated, only finite vals ...

            if dist_mat.shape[0] == 1:
                warnings.warn("I don't know how this can happen but there seems to be a"
                              " single an isolated difficult case ...", stacklevel=3)
                n_clusters = 1
                labels = torch.tensor([0])
            else:
                clusterer.fit(dist_mat)
                n_clusters = clusterer.n_clusters_
                labels = torch.from_numpy(clusterer.labels_)

            for c in range(n_clusters):
                in_cluster = labels == c
                feat_ix = alg_ix[in_cluster]
                if p_aggregation == 'sum':
                    p_agg = p_frame[in_cluster].sum()
                elif p_aggregation == 'max':
                    p_agg = p_frame[in_cluster].max()
                elif p_aggregation == 'pbinom_cdf':
                    z = fan_stat.binom_pdiverse(p_frame[in_cluster].view(-1))
                    p_agg = z[1:].sum()
                elif p_aggregation == 'pbinom_pdf':
                    z = fan_stat.binom_pdiverse(p_frame[in_cluster].view(-1))
                    p_agg = z[1]
                else:
                    raise ValueError

                p_out[i, 0, feat_ix[0]] = p_agg  # only set first element to some probability
                """Average the features."""
                feat_av = (feat_out[i, :, feat_ix] * p_frame[in_cluster]).sum(1) / p_frame[in_cluster].sum()
                feat_out[i, :, feat_ix] = feat_av.unsqueeze(1).repeat(1, in_cluster.sum())

        return p_out.reshape(p.size()), feat_out.reshape(features.size())

    def _cluster_mp(self, p: torch.Tensor, features: torch.Tensor):
        """
        Processes a batch in a multiprocessing fashion by splitting the batch into multiple smaller ones and forwards
        them through ._cluster_batch

        Args:
            p (torch.Tensor): detections
            features (torch.Tensor): features

        Returns:

        """

        p = p.cpu()
        features = features.cpu()
        batch_size = p.size(0)

        # split the tensors into smaller batches and multi-process them
        p_split = torch.split(p, math.ceil(batch_size / self.num_workers))
        feat_split = torch.split(features, math.ceil(batch_size / self.num_workers))

        args = zip(p_split, feat_split)

        results = Parallel(n_jobs=self.num_workers)(
            delayed(self._cluster_batch)(p_, f_) for p_, f_ in args)

        p_out_ = []
        feat_out_ = []
        for i in range(len(results)):
            p_out_.append(results[i][0])
            feat_out_.append(results[i][1])

        p_out = torch.cat(p_out_, dim=0)
        feat_out = torch.cat(feat_out_, dim=0)

        return p_out, feat_out

    def _forward_raw_impl(self, p, features):
        """
        Actual implementation.

        Args:
            p:
            features:

        Returns:

        """
        with torch.no_grad():
            """First init by an first threshold to get rid of all the nonsense"""
            p_clip = torch.zeros_like(p)
            is_above_svalue = p > self.raw_th
            p_clip[is_above_svalue] = p[is_above_svalue]
            # p_clip_rep = p_clip.repeat(1, features.size(1), 1, 1)  # repeated to access the features

            """Compute Local Mean Background"""
            if features.size(1) == 5:

                bg_out = features[:, [4]]
                bg_out = self._bg_calculator._mean_filter(bg_out).cpu()

            else:
                bg_out = None

            """Divide the set in easy (no neighbors) and set of predictions with adjacents"""
            binary_mask = torch.zeros_like(p_clip)
            binary_mask[p_clip > 0] = 1.

            # count neighbors
            self._neighbor_kernel = self._neighbor_kernel.type(binary_mask.dtype).to(binary_mask.device)
            count = torch.nn.functional.conv2d(binary_mask, self._neighbor_kernel, padding=1) * binary_mask

            # divide in easy and difficult set
            is_easy = count == 1
            is_easy_rep = is_easy.repeat(1, features.size(1), 1, 1)
            is_diff = count > 1
            is_diff_rep = is_diff.repeat(1, features.size(1), 1, 1)

            p_easy = torch.zeros_like(p_clip)
            p_diff = p_easy.clone()
            feat_easy = torch.zeros_like(features)
            feat_diff = feat_easy.clone()

            p_easy[is_easy] = p_clip[is_easy]
            feat_easy[is_easy_rep] = features[is_easy_rep]

            p_diff[is_diff] = p_clip[is_diff]
            feat_diff[is_diff_rep] = features[is_diff_rep]

            p_out = torch.zeros_like(p_clip).cpu()
            feat_out = torch.zeros_like(feat_diff).cpu()

            """Cluster the hard cases if they are consistent given euclidean affinity."""
            if self.num_workers == 0:
                p_out_diff, feat_out_diff = self._cluster_batch(p_diff.cpu(), feat_diff.cpu())
            else:
                p_out_diff, feat_out_diff = self._cluster_mp(p_diff, feat_diff)

            """Add the easy ones."""
            p_out[is_easy] = p_easy[is_easy].cpu()
            p_out[is_diff] = p_out_diff[is_diff].cpu()

            feat_out[is_easy_rep] = feat_easy[is_easy_rep].cpu()
            feat_out[is_diff_rep] = feat_out_diff[is_diff_rep].cpu()

            """Write the bg frame"""
            if features.size(1) == 5:
                feat_out[:, [4]] = bg_out

            return p_out, feat_out

    def _frame2emitter(self, p: torch.Tensor, features: torch.Tensor):
        """
        Convert frame based features to tensor based features (go frame from image world to emitter world)

        Args:
            p (torch.Tensor): detection channel
            features (torch.Tensor): features

        Returns:
            (torch.Tensor, torch.Tensor, torch.Tensor)

            feat_out: output features
            p_out: final probabilities
            batch_ix: batch index

        """
        is_pos = (p >= self.em_th).nonzero()  # is above threshold
        p_out = p[p >= self.em_th]

        # look up features
        feat_out = features[is_pos[:, 0], :, is_pos[:, 2], is_pos[:, 3]]

        # pick corresponding batch index
        batch_ix = torch.ones_like(p) * torch.arange(p.size(0), dtype=features.dtype).view(-1, 1, 1, 1)  # bookkeep
        batch_ix = batch_ix[is_pos[:, 0], :, is_pos[:, 2], is_pos[:, 3]]

        return feat_out, p_out, batch_ix.long()

    def forward(self, features: torch.Tensor):
        """
        Forward the feature map through the post processing and return an EmitterSet or a list of EmitterSets.
        For the input features we use the following convention:

            0 - Detection channel

            1 - Photon channel

            2 - 'x' channel

            3 - 'y' channel

            4 - 'z' channel

            5 - Background channel

        Expecting x and y channels in nano-metres.

        Args:
            features (torch.Tensor): Features of size :math:`(N, C, H, W)`

        Returns:
            EmitterSet or list of EmitterSets: Specified by return_format argument, EmitterSet in nano metres.

        """

        if features.dim() != 4:
            raise ValueError("Wrong dimensionality. Needs to be N x C x H x W.")

        if features.size(1) not in (5, 6):
            raise ValueError("Unsupported channel dimension.")

        p = features[:, [0], :, :]
        features = features[:, 1:, :, :]  # phot, x, y, z, bg

        p_out, feat_out = self._forward_raw_impl(p, features)

        feature_list, prob_final, frame_ix = self._frame2emitter(p_out, feat_out)
        frame_ix = frame_ix.squeeze()

        em = EmitterSet(xyz=feature_list[:, 1:4], phot=feature_list[:, 0], frame_ix=frame_ix,
                        prob=prob_final, bg=feature_list[:, 4],
                        xy_unit=self.xy_unit, px_size=self.px_size)

        return self._return_as_type(em, ix_low=0, ix_high=features.size(0))


class Offset2Coordinate:
    """
    Convert sub-pixel pointers to absolute coordinates.
    """

    def __init__(self, xextent: tuple, yextent: tuple, img_shape: tuple):
        """

        Args:
            xextent (tuple): extent in x
            yextent (tuple): extent in y
            img_shape (tuple): image shape
        """

        off_psf = UnifiedEmbeddingTarget(xextent=xextent,
                                         yextent=yextent,
                                         img_shape=img_shape, roi_size=1)

        xv, yv = torch.meshgrid([off_psf._bin_ctr_x, off_psf._bin_ctr_y])
        self._x_mesh = xv.unsqueeze(0)
        self._y_mesh = yv.unsqueeze(0)

    def _subpx_to_absolute(self, x_offset, y_offset):
        """
        Convert subpixel pointers to absolute coordinates. Actual implementation

        Args:
            x_offset:
            y_offset:

        Returns:

        """
        batch_size = x_offset.size(0)
        x_coord = self._x_mesh.repeat(batch_size, 1, 1).to(x_offset.device) + x_offset
        y_coord = self._y_mesh.repeat(batch_size, 1, 1).to(y_offset.device) + y_offset
        return x_coord, y_coord

    @staticmethod
    def parse(param):
        return Offset2Coordinate(param.Simulation.psf_extent[0],
                                 param.Simulation.psf_extent[1],
                                 param.Simulation.img_size)

    def forward(self, x: torch.Tensor):
        """
        Forward frames through post-processor.

        Args:
            x (torch.Tensor): features to be converted; expected shape :math:`(N, C, H, W)`

        Returns:

        """

        if x.dim() != 4:
            raise ValueError("Wrong dimensionality. Needs to be N x C x H x W.")

        """Convert the channel values to coordinates"""
        x_coord, y_coord = self._subpx_to_absolute(x[:, 2], x[:, 3])

        output_converted = x.clone()
        output_converted[:, 2] = x_coord
        output_converted[:, 3] = y_coord

        return output_converted


@deprecated(version="0.1", reason="Not used. Needs tests if you want to use it.")
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


@deprecated(version="0.1", reason="Not used. Needs tests if you want to use it.")
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


@deprecated(version="0.1", reason="Not used. Needs tests if you want to use it.")
class ConnectedComponents:
    def __init__(self, svalue_th=0, connectivity=2):
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

        """Set all values under the single value threshold to 0."""
        p[p < self.svalue_th] = 0.

        cluster_ix = torch.zeros_like(p)

        for i in range(p.size(0)):
            c_ix, _ = label(p[i].numpy(), self.kernel)
            cluster_ix[i] = torch.from_numpy(c_ix)

        if out_hw:
            return cluster_ix.squeeze(0)
        else:
            return cluster_ix


@deprecated(version="0.1", reason="Not used. Needs tests if you want to use it.")
class SpeiserPost:

    def __init__(self, svalue_th=0.3, sep_th=0.6, out_format='emitters', out_th=0.7, p_agg='max'):
        """

        :param svalue_th: single value threshold
        :param sep_th: threshold when to assume that we have 2 emitters
        :param out_format: either 'emitters' or 'image'. If 'emitter' we output instance of EmitterSet, if 'frames' we output post_processed frames.
        :param out_th: final threshold
        """
        self.svalue_th = svalue_th
        self.sep_th = sep_th
        self.out_format = out_format
        self.out_th = out_th
        self._p_agg = p_agg

    @staticmethod
    def parse(param: dict):
        return SpeiserPost(param['PostProcessing']['single_val_th'],
                           param['PostProcessing']['total_th'],
                           'emitters_framewise')

    @staticmethod
    def frame_to_emitter(prob, features, threshold):
        pass
        # Todo

    def forward_(self, p, features):
        """
        :param p: N x H x W probability map
        :param features: N x C x H x W features
        :return: feature averages N x (1 + C) x H x W final probabilities plus features
        """
        with torch.no_grad():
            diag = 0
            p_ = p.clone()
            features = features.clone()

            # probability values > 0.3 are regarded as possible locations
            p_clip = torch.where(p > self.svalue_th, p, torch.zeros_like(p))[:, None]

            # localize maximum values within a 3x3 patch
            pool = torch.nn.functional.max_pool2d(p_clip, 3, 1, padding=1)
            max_mask1 = torch.eq(p[:, None], pool).float()

            # Add probability values from the 4 adjacent pixels
            filt = torch.tensor([[diag, 1, diag], [1, 1, 1], [diag, 1, diag]]).view(1, 1, 3, 3). \
                type(features.dtype). \
                to(features.device)
            conv = torch.nn.functional.conv2d(p[:, None], filt, padding=1)
            p_ps1 = max_mask1 * conv

            """In order do be able to identify two fluorophores in adjacent pixels we look for 
            probablity values > 0.6 that are not part of the first mask."""
            p_ = p * (1 - max_mask1[:, 0])
            # p_clip = torch.where(p_ > self.sep_th, p_, torch.zeros_like(p_))[:, None]
            max_mask2 = torch.where(p_ > self.sep_th, torch.ones_like(p_), torch.zeros_like(p_))[:, None]
            """
            If we still have now adjacent pixels with a prob bigger than twice the seperation threshold, 
            throw it away. One might think of a more sophisticated ambiguity handling.
            """
            # ToDo: I don't know whether this is correct!
            count_mask2 = torch.nn.functional.conv2d(max_mask2, filt, padding=1)
            ambig_mask = torch.ones_like(count_mask2)
            ambig_mask[ambig_mask >= 2 * self.sep_th] = 0.
            max_mask2 *= ambig_mask

            p_ps2 = max_mask2 * conv

            """This is our final clustered probablity which we then threshold (normally > 0.7) 
            to get our final discrete locations."""
            if self._p_agg == 'sum':
                p_ps = p_ps1 + p_ps2
            if self._p_agg == 'max':
                p_ps = torch.max(p_ps1, p_ps2)

            max_mask = torch.clamp(max_mask1 + max_mask2, 0, 1)

            mult_1 = max_mask1 / p_ps1
            mult_1[torch.isnan(mult_1)] = 0
            mult_2 = max_mask2 / p_ps2
            mult_2[torch.isnan(mult_2)] = 0

            feat_out = torch.zeros_like(features)
            for i in range(features.size(1)):
                feature_mid = features[:, i] * p
                feat_conv1 = torch.nn.functional.conv2d((feature_mid * (1 - max_mask2[:, 0]))[:, None], filt, padding=1)
                feat_conv2 = torch.nn.functional.conv2d((feature_mid * (1 - max_mask1[:, 0]))[:, None], filt, padding=1)

                feat_out[:, [i]] = feat_conv1 * mult_1 + feat_conv2 * mult_2

            feat_out[torch.isnan(feat_out)] = 0

        """Output """
        combined_output = torch.cat((p_ps, feat_out), dim=1)

        return combined_output

    def forward(self, features):
        """
        Wrapper method calling forward_masked which is the actual implementation.

        :param features: NCHW
        :return: feature averages N x C x H x W if self.out_format == frames,
            list of EmitterSets if self.out_format == 'emitters'
        """
        post_frames = self.forward_(features[:, 0], features[:, 1:]).cpu()
        is_above_out_th = (post_frames[:, [0], :, :] > self.out_th)

        post_frames = post_frames * is_above_out_th.type(post_frames.dtype)

        batch_size = post_frames.shape[0]

        """Output according to format as specified."""
        if self.out_format == 'frames':
            return post_frames

        elif self.out_format[:8] == 'emitters':
            is_above_out_th.squeeze_(1)
            frame_ix = torch.ones_like(post_frames[:, 0, :, :]) * \
                       torch.arange(batch_size, dtype=post_frames.dtype).view(-1, 1, 1, 1)
            frame_ix = frame_ix[:, 0, :, :][is_above_out_th]
            p_map = post_frames[:, 0, :, :][is_above_out_th]
            phot_map = post_frames[:, 1, :, :][is_above_out_th]
            x_map = post_frames[:, 2, :, :][is_above_out_th]
            y_map = post_frames[:, 3, :, :][is_above_out_th]
            z_map = post_frames[:, 4, :, :][is_above_out_th]
            xyz = torch.cat((
                x_map.unsqueeze(1),
                y_map.unsqueeze(1),
                z_map.unsqueeze(1)
            ), 1)

            em = EmitterSet(xyz, phot_map, frame_ix, prob=p_map)
            if self.out_format[8:] == '_batch':
                return em
            elif self.out_format[8:] == '_framewise':
                return em.split_in_frames(0, batch_size - 1)


@deprecated(version="0.1", reason="Not used. Needs tests if you want to use it.")
class CC5ChModel(ConnectedComponents):
    """Connected components on 5 channel model."""

    def __init__(self, prob_th, svalue_th=0, connectivity=2):
        super().__init__(svalue_th, connectivity)
        self.prob_th = prob_th

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
            ix_above_prob_th = pi >= self.prob_th
            phot_red = feat_i[:, 1]
            xyz = torch.cat((
                feat_i[:, 2].unsqueeze(1),
                feat_i[:, 3].unsqueeze(1),
                feat_i[:, 4].unsqueeze(1)), 1)

            em = EmitterSet(xyz[ix_above_prob_th],
                            phot_red[ix_above_prob_th],
                            frame_ix=(i * torch.ones_like(phot_red[ix_above_prob_th])))

            emitter_sets[i] = em
        if red_batch:
            return emitter_sets[0]
        else:
            return emitter_sets


@deprecated(version="0.1", reason="Not used. Needs tests if you want to use it.")
class CCDirectPMap(CC5ChModel):
    def __init__(self, extent, img_shape, prob_th, svalue_th=0, connectivity=2):
        """

        :param photon_threshold: minimum total value of summmed output
        :param extent:
        :param clusterer:
        :param single_value_threshold:
        :param connectivity:
        """
        super().__init__(svalue_th, connectivity)
        self.extent = extent
        self.prob_th = prob_th

        self.clusterer = label
        self.matrix_extent = None
        self.connectivity = connectivity

        self.offset2coordinate = Offset2Coordinate(extent[0], extent[1], img_shape)

        if self.connectivity == 2:
            self.kernel = np.ones((3, 3))
        elif self.connectivity == 1:
            self.kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])

    def forward(self, x):
        """
        Forward a batch of frames through connected components. Must only contain one channel.

        :param x: 2D frame, or 4D batch of 1 channel frames.
        :return: (instance of emitterset)
        """

        # loop over all batch elements
        if not (x.dim() == 3 or x.dim() == 4):
            raise ValueError("Input must be C x H x W or N x C x H x W.")
        elif x.dim() == 3:
            x = x.unsquueze(0)

        """Generate a pseudo offset (with 0zeros) to use the present CC5ch model."""
        x_pseudo = torch.zeros((x.size(0), 5, x.size(2), x.size(3)))
        x_pseudo[:, 0] = x

        """Run the pseudo offsets through the Offset2Coordinate"""
        x_pseudo = self.offset2coordinate.forward(x_pseudo)

        """Run the super().forward as we are now in the same stiuation as for the 5 channel offset model."""
        return super().forward(x_pseudo)


@deprecated(version="0.1", reason="Not used? Write a test first.")
def crlb_squared_distance(X, Y, XCrlb, YCrlb):
    """
    Computes the CRLB (Cramer Rao Lower Bound) weighted distances between the vectors X and Y
    :param X:
    :param Y:
    :param XCrlb:
    :param YCrlb:
    :return: squarred distance in units of CRLB
    """
    dist = (X - Y) ** 2 / (XCrlb ** 2 + YCrlb ** 2)
    dist = dist.sum(1)
    return dist
