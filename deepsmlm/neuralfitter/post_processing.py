import math
import warnings
from abc import ABC, abstractmethod  # abstract class

import torch
from joblib import Parallel, delayed
from sklearn.cluster import AgglomerativeClustering

import deepsmlm.simulation.background
from deepsmlm.evaluation import match_emittersets
from deepsmlm.generic.emitter import EmitterSet, EmptyEmitterSet
from deepsmlm.neuralfitter.target_generator import UnifiedEmbeddingTarget


class PostProcessing(ABC):
    _return_types = ('batch-set', 'frame-set')

    def __init__(self, xy_unit, px_size, return_format: str):
        """

        Args:
            return_format (str): return format of forward function. Must be 'batch-set', 'frame-set'. If 'batch-set'
            one instance of EmitterSet will be returned per forward call, if 'frame-set' a tuple of EmitterSet one
            per frame will be returned
            sanity_check (bool): perform sanity check
        """

        super().__init__()
        self.xy_unit = xy_unit
        self.px_size = px_size
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

    def __init__(self, xy_unit=None, px_size=None, return_format='batch-set'):
        super().__init__(xy_unit=xy_unit, px_size=px_size, return_format=return_format)

    def forward(self, x: torch.Tensor):
        """

        Args:
            x (torch.Tensor): any input tensor where the first dim is the batch-dim.

        Returns:
            EmptyEmitterSet: An empty EmitterSet

        """

        em = EmptyEmitterSet(xy_unit=self.xy_unit, px_size=self.px_size)
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
                 skip_th: (None, float) = None, return_format='batch-set', sanity_check=True):
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
            skip_th: relative fraction of the detection output to be on to skip post_processing.
                This is useful during training when the network has not yet converged and major parts of the
                detection output is white (i.e. non sparse detections).
            return_format:
            sanity_check:
        """
        super().__init__(xy_unit=xy_unit, px_size=px_size, return_format=return_format)

        self.raw_th = raw_th
        self.em_th = em_th
        self.p_aggregation = p_aggregation
        self.match_dims = match_dims
        self.num_workers = num_workers
        self.skip_th = skip_th

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

    @classmethod
    def parse(cls, param, **kwargs):
        """
        Return an instance of this post-processing as specified by the parameters

        Args:
            param:

        Returns:
            ConsistencyPostProcessing

        """
        return cls(raw_th=param.PostProcessing.single_val_th, em_th=param.PostProcessing.total_th,
                   xy_unit='px', px_size=param.Camera.px_size,
                   img_shape=param.TestSet.img_size,
                   ax_th=param.PostProcessing.ax_th, vol_th=param.PostProcessing.vol_th,
                   lat_th=param.PostProcessing.lat_th, match_dims=param.PostProcessing.match_dims,
                   return_format='batch-set', **kwargs)

    def sanity_check(self):
        """
        Performs some sanity checks. Part of the constructor; useful if you modify attributes later on and want to
        double check.

        """

        super().sanity_check()

        if self.p_aggregation not in self._p_aggregations:
            raise ValueError("Unsupported probability aggregation type.")

    def skip_if(self, x):
        if x.dim() != 4:
            raise ValueError("Unsupported dim.")

        if self.skip_th is not None and (x[:, 0] >= self.raw_th).sum() > self.skip_th * x[:, 0].numel():
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
                    z = binom_pdiverse(p_frame[in_cluster].view(-1))
                    p_agg = z[1:].sum()
                elif p_aggregation == 'pbinom_pdf':
                    z = binom_pdiverse(p_frame[in_cluster].view(-1))
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

        if self.skip_if(features):
            return EmptyEmitterSet(xy_unit=self.xy_unit, px_size=self.px_size)

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
        return Offset2Coordinate(param.TestSet.frame_extent[0],
                                 param.TestSet.frame_extent[1],
                                 param.TestSet.img_size)

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


def binom_pdiverse(p):
    """
    binomial probability but unequal probabilities
    Args:
        p: (torch.Tensor) of probabilities

    Returns:
        z: (torch.Tensor) vector of probabilities with length p.size() + 1

    """
    n = p.size(0) + 1
    z = torch.zeros((n,))
    z[0] = 1

    for u in p:
        z = (1 - u) * z + u * torch.cat((torch.zeros((1,)), z[:-1]))

    return z
