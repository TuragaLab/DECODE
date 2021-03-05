import warnings
from abc import ABC, abstractmethod  # abstract class
from typing import Union, Callable

import scipy
import torch
from deprecated import deprecated
from sklearn.cluster import AgglomerativeClustering

import decode.simulation.background
from decode.evaluation import match_emittersets
from decode.generic.emitter import EmitterSet, EmptyEmitterSet
from decode.neuralfitter.utils.probability import binom_pdiverse


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

    @deprecated(reason="Not of interest for the post-processing.", version="0.1.dev")
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
    def forward(self, x: torch.Tensor) -> (EmitterSet, list):
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
    The 'No' Post-Processing post-processing. Will always return an empty EmitterSet.

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

        return EmptyEmitterSet(xy_unit=self.xy_unit, px_size=self.px_size)


class LookUpPostProcessing(PostProcessing):
    """
    Simple post-processing in which we threshold the probability output (raw threshold) and then look-up the features
    in the respective channels.

    """

    def __init__(self, raw_th: float, xy_unit: str, px_size=None,
                 pphotxyzbg_mapping: Union[list, tuple] = (0, 1, 2, 3, 4, -1),
                 photxyz_sigma_mapping: Union[list, tuple, None] = (5, 6, 7, 8)):
        """

        Args:
            raw_th: initial raw threshold
            xy_unit: xy unit unit
            px_size: pixel size
            pphotxyzbg_mapping: channel index mapping of detection (p), photon, x, y, z, bg
        """
        super().__init__(xy_unit=xy_unit, px_size=px_size, return_format='batch-set')

        self.raw_th = raw_th
        self.pphotxyzbg_mapping = pphotxyzbg_mapping
        self.photxyz_sigma_mapping = photxyz_sigma_mapping

        assert len(self.pphotxyzbg_mapping) == 6, "Wrong length of mapping."
        if self.photxyz_sigma_mapping is not None:
            assert len(self.photxyz_sigma_mapping) == 4, "Wrong length of sigma mapping."

    def _filter(self, detection) -> torch.BoolTensor:
        """

        Args:
            detection: any tensor that should be thresholded

        Returns:
            boolean with active px

        """

        return detection >= self.raw_th

    @staticmethod
    def _lookup_features(features: torch.Tensor, active_px: torch.Tensor) -> tuple:
        """

        Args:
            features: size :math:`(N, C, H, W)`
            active_px: size :math:`(N, H, W)`

        Returns:
            torch.Tensor: batch-ix, size :math: `M`
            torch.Tensor: extracted features size :math:`(C, M)`

        """

        assert features.dim() == 4
        assert active_px.dim() == features.dim() - 1

        batch_ix = active_px.nonzero(as_tuple=False)[:, 0]
        features_active = features.permute(1, 0, 2, 3)[:, active_px]

        return batch_ix, features_active

    def forward(self, x: torch.Tensor) -> EmitterSet:
        """
        Forward model output tensor through post-processing and return EmitterSet. Will include sigma values in
        EmitterSet if mapping was provided initially.

        Args:
            x: model output

        Returns:
            EmitterSet

        """
        """Reorder features channel-wise."""
        x_mapped = x[:, self.pphotxyzbg_mapping]

        """Filter"""
        active_px = self._filter(x_mapped[:, 0])  # 0th ch. is detection channel
        prob = x_mapped[:, 0][active_px]

        """Look-Up in channels"""
        frame_ix, features = self._lookup_features(x_mapped[:, 1:], active_px)

        """Return EmitterSet"""
        xyz = features[1:4].transpose(0, 1)

        """If sigma mapping is present, get those values as well."""
        if self.photxyz_sigma_mapping is not None:
            sigma = x[:, self.photxyz_sigma_mapping]
            _, features_sigma = self._lookup_features(sigma, active_px)

            xyz_sigma = features_sigma[1:4].transpose(0, 1).cpu()
            phot_sigma = features_sigma[0].cpu()
        else:
            xyz_sigma = None
            phot_sigma = None

        return EmitterSet(xyz=xyz.cpu(), frame_ix=frame_ix.cpu(), phot=features[0, :].cpu(),
                          xyz_sig=xyz_sigma, phot_sig=phot_sigma, bg_sig=None,
                          bg=features[4, :].cpu() if features.size(0) == 5 else None,
                          prob=prob.cpu(), xy_unit=self.xy_unit, px_size=self.px_size)


class SpatialIntegration(LookUpPostProcessing):
    """
    Spatial Integration post processing.
    """

    _p_aggregations = ('sum', 'norm_sum')  # , 'max', 'pbinom_cdf', 'pbinom_pdf')
    _split_th = 0.6

    def __init__(self, raw_th: float, xy_unit: str, px_size=None,
                 pphotxyzbg_mapping: Union[list, tuple] = (0, 1, 2, 3, 4, -1),
                 photxyz_sigma_mapping: Union[list, tuple, None] = (5, 6, 7, 8),
                 p_aggregation: Union[str, Callable] = 'norm_sum'):
        """

        Args:
            raw_th: probability threshold from where detections are considered
            xy_unit: unit of the xy coordinates
            px_size: pixel size
            pphotxyzbg_mapping: channel index mapping
            photxyz_sigma_mapping: channel index mapping of sigma channels
            p_aggregation: aggreation method to aggregate probabilities. can be 'sum', 'max', 'norm_sum'
        """
        super().__init__(raw_th=raw_th, xy_unit=xy_unit, px_size=px_size,
                         pphotxyzbg_mapping=pphotxyzbg_mapping,
                         photxyz_sigma_mapping=photxyz_sigma_mapping)

        self.p_aggregation = self.set_p_aggregation(p_aggregation)

    def forward(self, x: torch.Tensor) -> EmitterSet:
        x[:, 0] = self._nms(x[:, 0], self.p_aggregation, self.raw_th, self._split_th)

        return super().forward(x)

    @staticmethod
    def _nms(p: torch.Tensor, p_aggregation, raw_th, split_th) -> torch.Tensor:
        """
        Non-Maximum Suppresion

        Args:
            p:

        """

        with torch.no_grad():
            p_copy = p.clone()

            """Probability values > 0.3 are regarded as possible locations"""
            p_clip = torch.where(p > raw_th, p, torch.zeros_like(p))[:, None]

            """localize maximum values within a 3x3 patch"""
            pool = torch.nn.functional.max_pool2d(p_clip, 3, 1, padding=1)
            max_mask1 = torch.eq(p[:, None], pool).float()

            """Add probability values from the 4 adjacent pixels"""
            diag = 0.  # 1/np.sqrt(2)
            filt = torch.tensor([[diag, 1., diag], [1, 1, 1], [diag, 1, diag]]).unsqueeze(0).unsqueeze(0).to(p.device)
            conv = torch.nn.functional.conv2d(p[:, None], filt, padding=1)
            p_ps1 = max_mask1 * conv

            """
            In order do be able to identify two fluorophores in adjacent pixels we look for
            probablity values > 0.6 that are not part of the first mask
            """
            p_copy *= (1 - max_mask1[:, 0])
            # p_clip = torch.where(p_copy > split_th, p_copy, torch.zeros_like(p_copy))[:, None]
            max_mask2 = torch.where(p_copy > split_th, torch.ones_like(p_copy), torch.zeros_like(p_copy))[:, None]
            p_ps2 = max_mask2 * conv

            """This is our final clustered probablity which we then threshold (normally > 0.7)
            to get our final discrete locations"""
            p_ps = p_aggregation(p_ps1, p_ps2)
            assert p_ps.size(1) == 1

            return p_ps.squeeze(1)

    @classmethod
    def set_p_aggregation(cls, p_aggr: Union[str, Callable]) -> Callable:
        """
        Sets the p_aggregation by string or callable. Return s Callable

        Args:
            p_aggr: probability aggregation

        """

        if isinstance(p_aggr, str):

            if p_aggr == 'sum':
                return torch.add
            elif p_aggr == 'max':
                return torch.max
            elif p_aggr == 'norm_sum':
                def norm_sum(*args):
                    return torch.clamp(torch.add(*args), 0., 1.)

                return norm_sum
            else:
                raise ValueError

        else:
            return p_aggr


class ConsistencyPostprocessing(PostProcessing):
    """
    PostProcessing implementation that divides the output in hard and easy samples. Easy samples are predictions in
    which we have a single one hot pixel in the detection channel, hard samples are pixels in the detection channel
    where the adjacent pixels are also active (i.e. above a certain initial threshold).
    """
    _p_aggregations = ('sum', 'max', 'pbinom_cdf', 'pbinom_pdf')
    _xy_unit = 'nm'

    def __init__(self, *, raw_th, em_th, xy_unit: str, img_shape, ax_th=None, vol_th=None, lat_th=None,
                 p_aggregation='pbinom_cdf', px_size=None, match_dims=2, diag=0, pphotxyzbg_mapping=[0, 1, 2, 3, 4, -1],
                 num_workers=0, skip_th: (None, float) = None, return_format='batch-set', sanity_check=True):
        """

        Args:
            pphotxyzbg_mapping:
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

        self.pphotxyzbg_mapping = pphotxyzbg_mapping

        self._filter = match_emittersets.GreedyHungarianMatching(match_dims=match_dims, dist_lat=lat_th,
                                                                 dist_ax=ax_th, dist_vol=vol_th).filter

        self._bg_calculator = decode.simulation.background.BgPerEmitterFromBgFrame(filter_size=13, xextent=(0., 1.),
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
        return cls(raw_th=param.PostProcessingParam.single_val_th, em_th=param.PostProcessingParam.total_th,
                   xy_unit='px', px_size=param.Camera.px_size,
                   img_shape=param.TestSet.img_size,
                   ax_th=param.PostProcessingParam.ax_th, vol_th=param.PostProcessingParam.vol_th,
                   lat_th=param.PostProcessingParam.lat_th, match_dims=param.PostProcessingParam.match_dims,
                   return_format='batch-set', **kwargs)

    def sanity_check(self):
        """
        Performs some sanity checks. Part of the constructor; useful if you modify attributes later on and want to
        double check.

        """

        super().sanity_check()

        if self.p_aggregation not in self._p_aggregations:
            raise ValueError("Unsupported probability aggregation type.")

        if len(self.pphotxyzbg_mapping) != 6:
            raise ValueError(f"Wrong channel mapping length.")

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
                dist_mat = torch.pdist(f_frame[:, 1:3])
            elif self.match_dims == 3:
                dist_mat = torch.pdist(f_frame[:, 1:4])
            else:
                raise ValueError

            dist_mat = torch.from_numpy(scipy.spatial.distance.squareform(dist_mat))
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
                raise NotImplementedError

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

        features = features[:, self.pphotxyzbg_mapping]  # change channel order if needed

        p = features[:, [0], :, :]
        features = features[:, 1:, :, :]  # phot, x, y, z, bg

        p_out, feat_out = self._forward_raw_impl(p, features)

        feature_list, prob_final, frame_ix = self._frame2emitter(p_out, feat_out)
        frame_ix = frame_ix.squeeze()

        return EmitterSet(xyz=feature_list[:, 1:4], phot=feature_list[:, 0], frame_ix=frame_ix,
                          prob=prob_final, bg=feature_list[:, 4],
                          xy_unit=self.xy_unit, px_size=self.px_size)
