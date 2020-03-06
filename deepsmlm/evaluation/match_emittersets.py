from abc import ABC, abstractmethod
from collections import namedtuple
from functools import partial
from deprecated import deprecated
import warnings
import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors
from scipy.optimize import linear_sum_assignment
from deepsmlm.generic import emitter as emitter


class MatcherABC(ABC):
    """
    Abstract match class.
    """

    _return_match = namedtuple('return_match', ['tp', 'fp', 'fn', 'tp_match'])  # return-type as namedtuple

    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, output: emitter.EmitterSet, target: emitter.EmitterSet) -> _return_match:
        """
        All implementations must implement forward method.

        Args:
            output:
            target:

        Returns:
            tp: true positives
            fp: false positives
            fn: false negatives
            tp_match: ground truths that have been matched to the true positives
        """
        raise NotImplementedError


class GreedyHungarianMatching(MatcherABC):
    """
    Matching emitters in a greedy 'hungarian' fashion, by using best first search.

    Attributes:

    """
    def __init__(self, *, match_dims: int, dist_ax: float = None, dist_lat: float = None, dist_vol: float = None):
        """
        Initialise "Greedy Hungarian Matching". Incorporates some rule-out thresholds

        Args:
            match_dims (int): match in 2D or 3D
            dist_lat: lateral tolerance radius
            dist_ax: axial tolerance threshold
            dist_vol: volumetric tolerance radius
        """

        self.match_dims = match_dims
        self.dist_ax = dist_ax
        self.dist_lat = dist_lat
        self.dist_vol = dist_vol

        """Sanity checks"""
        if self.match_dims not in (2, 3):
            raise ValueError("Not supported match dimensionality.")

        if self.dist_lat is not None and self.dist_ax is not None and self.dist_vol is not None:
            warnings.warn("You specified a lateral, axial and volumetric threshold. "
                          "While this is allowed; are you sure?")

        if self.dist_lat is None and self.dist_ax is None and self.dist_vol is None:
            warnings.warn("You specified neither a lateral, axial nor volumetric threshold. Are you sure about this?")

    def _filter(self, xyz_out, xyz_tar) -> torch.Tensor:
        """
        Filter kernel to rule out unwanted matches. Batch implemented.

        Args:
            xyz_out: output coordinates B x N x 3
            xyz_tar: target coordinates B x M x 3

        Returns:
            filter_mask (torch.Tensor): boolean of size B x N x M
        """
        assert xyz_out.size(0) == xyz_tar.size(0)
        filter_mask = torch.ones((xyz_out.size(0), xyz_out.size(1), xyz_tar.size(1))).bool()

        if self.dist_lat is not None:
            dist_mat = torch.cdist(xyz_out[:, :, :2], xyz_tar[:, :, :2], p=2)
            filter_mask[dist_mat > self.dist_lat ** 2] = 0

        if self.dist_ax is not None:
            dist_mat = torch.cdist(xyz_out[:, :, [2]], xyz_tar[:, :, [2]], p=2)
            filter_mask[dist_mat > self.dist_ax ** 2] = 0

        if self.dist_vol is not None:
            dist_mat = torch.cdist(xyz_out, xyz_tar, p=2)
            filter_mask[dist_mat > self.dist_vol ** 2] = 0

        return filter_mask

    @staticmethod
    def _rule_out_kernel(dists):
        """
        Kernel which goes through the distance matrix, picks shortest distance and assign match

        Args:
            dists: distance matrix

        Returns:

        """
        if dists.numel() == 0:
            return torch.zeros((0, )).int(), torch.zeros((0, )).int()

        dists_ = dists.clone()

        match_list = []
        while not (dists_ == float('inf')).all():
            ix = np.unravel_index(dists_.argmin(), dists_.shape)
            dists_[ix[0]] = float('inf')
            dists_[:, ix[1]] = float('inf')

            match_list.append(ix)

        if match_list.__len__() >= 1:
            assignment = torch.tensor(match_list)
        else:
            assignment = torch.zeros((0, 2)).int()

        return assignment[:, 0], assignment[:, 1]

    def _match_kernel(self, xyz_out, xyz_tar, filter_mask):
        """

        Args:
            xyz_out: N x 3  - no batch implementation currently
            xyz_tar: M x 3 - no batch implementation currently
            filter_mask: N x M - not batched

        Returns:
            out_ind: index for xyz_out

        """
        if self.match_dims == 2:
            dist_mat = torch.cdist(xyz_out[None, :, :2], xyz_tar[None, :, :2], p=2)
        elif self.match_dims == 3:
            dist_mat = torch.cdist(xyz_out[None, :, :], xyz_tar[None, :, :], p=2)
        else:
            raise ValueError

        dist_mat[~filter_mask.unsqueeze(0)] = float('inf')  # rule out matches by filter
        tp_ix, tp_match_ix = self._rule_out_kernel(dist_mat)

        return tp_ix, tp_match_ix

    def forward(self):
        raise NotImplementedError

@deprecated
class GreedyHungarianMatchingDepr(MatcherABC):
    """
    Matching emitters in a greedy 'hungarian' fashion, by using best first search.

    Attributes:

    """

    def __init__(self, *, match_dims: int,
                 dist_ax: float = None, dist_vol: float = None, dist_lat: float = None):
        """
        Initialise "Greedy Hungarian Matching". If you specify dist_lat and dist_ax you can specify whether matching
        should be performed in 3D or in 2D. Otherwise it will be automatically determined. If you specify 2D while
        providing a lateral and an axial threshold, the axial threshold will only be used to exclude matchs that are
        too far off in the axial direction.

        Args:
            match_dims (int): match in 2D or 3D
            dist_lat: lateral tolerance radius
            dist_ax: axial tolerance threshold
            dist_vol: volumetric tolerance radius
        """
        super().__init__()

        self._dist_thresh = None
        self._rule_out_thresh = None
        self._match_dims = None
        self._cdist_kernel = None

        """Some safety checks."""
        if ((dist_lat is not None) and (dist_vol is not None)) or ((dist_lat is None) and (dist_vol is None)):
            raise ValueError("You need to specify exactly exclusively either dist_lat or dist_vol.")

        if (dist_ax is not None) and (dist_vol is not None):
            raise ValueError("You can not specify dist_ax and dist_vol.")



        """Set the threshold and apropriate match_dim logic."""
        if dist_lat is not None and dist_ax is None:
            self._dist_thresh = dist_lat
            self._match_dims = 2
        elif dist_lat is not None and dist_ax is not None:
            self._dist_thresh = dist_lat
            self.rule_out_thresh = dist_ax  # kick out things which are too far off in z.

            # either match in 2D or 3D, this can be both.
            if match_dims is not None:
                if match_dims in (2, 3):
                    self._match_dims = match_dims
                else:
                    raise ValueError("Match dimension not allowed.")
            else:
                raise ValueError("You need to specify whether you want to match in 2D or 3D, when specifying both"
                                 "dist_lat and dist_ax")

        elif dist_vol is not None:
            self._dist_thresh = dist_vol
            self._match_dims = 3

        self._cdist_kernel = partial(torch.cdist, p=2)  # does not take the square root

    @staticmethod
    def parse(param):
        return GreedyHungarianMatching(match_dims=param.Evaluation.match_dims, dist_ax=param.Evaluation.dist_ax,
                                       dist_vol=param.Evaluation.dist_vol, dist_lat=param.Evaluation.dist_lat)

    @staticmethod
    def rule_out_dist_match(dists, threshold):
        """
        Kernel which goes through the distance matrix, picks shortest distance and assign match
         until a threshold is reached.

        Args:
            dists: distance matrix
            threshold: threshold until match

        Returns:

        """
        dists_ = dists.clone()

        match_list = []
        while dists_.min() < threshold:
            ix = np.unravel_index(dists_.argmin(), dists_.shape)
            dists_[ix[0]] = float('inf')
            dists_[:, ix[1]] = float('inf')

            match_list.append(ix)
        if match_list.__len__() >= 1:
            return torch.tensor(match_list)
        else:
            return torch.zeros((0, 2)).int()

    def assign_kernel(self, out, tar):
        """
        Assigns out and tar, blind to a frame index. Therefore, split in frames beforehand
        :param out:
        :param tar:
        :return:
        """
        """If no emitter has been found, all are false negatives. No tp, no fp."""
        if out.__len__() == 0:
            tp, fp, tp_match = emitter.EmptyEmitterSet(), emitter.EmptyEmitterSet(), emitter.EmptyEmitterSet()
            fn = tar
            return tp, fp, fn, tp_match

        """If there were no positives, no tp, no fn, all fp, no match."""
        if tar.__len__() == 0:
            tp, fn, tp_match = emitter.EmptyEmitterSet(), emitter.EmptyEmitterSet(), emitter.EmptyEmitterSet()
            fp = out
            return tp, fp, fn, tp_match

        if self._match_dims == 2:
            dists = self._cdist_kernel(out.xyz[:, :2], tar.xyz[:, :2])
        elif self._match_dims == 3:
            dists = self._cdist_kernel(out.xyz, tar.xyz)

        if self._rule_out_thresh is not None:
            dists_ax = self._cdist_kernel(out.xyz[:, [2]], tar.xyz[:, [2]])
            dists[dists_ax > self._rule_out_thresh] = float('inf')

        match_ix = self.rule_out_dist_match(dists, self._dist_thresh).numpy()
        all_ix_out = np.arange(out.__len__())
        all_ix_tar = np.arange(tar.__len__())

        tp = out.get_subset(match_ix[:, 0])
        tp_match = tar.get_subset(match_ix[:, 1])

        fp_ix = torch.from_numpy(np.setdiff1d(all_ix_out, match_ix[:, 0]))
        fn_ix = torch.from_numpy(np.setdiff1d(all_ix_tar, match_ix[:, 1]))

        fp = out.get_subset(fp_ix)
        fn = tar.get_subset(fn_ix)

        return tp, fp, fn, tp_match

    def forward(self, output: emitter.EmitterSet, target: emitter.EmitterSet):
        """
        Matches two sets of emitters.

        Args:
            output: predicted localisations
            target: ground truth localisations

        Returns:
            tp: true positives
            fp: false positives
            fn: false negatives
            tp_match  gt localisations that were considered found (by means of this matching)
        """

        """Setup split in frames. Determine the frame range automatically so as to cover everything."""
        if output.frame_ix.min() < target.frame_ix.min():
            frame_low = output.frame_ix.min().item()
        else:
            frame_low = target.frame_ix.min().item()

        if output.frame_ix.min() > target.frame_ix.max():
            frame_high = output.frame_ix.max().item()
        else:
            frame_high = target.frame_ix.max().item()

        out_pframe = output.split_in_frames(frame_low, frame_high)
        tar_pframe = target.split_in_frames(frame_low, frame_high)

        tpl, fpl, fnl, tpml = [], [], [], []  # true positive list, false positive list, false neg. ...

        """Assign the emitters framewise"""
        for i in range(out_pframe.__len__()):
            tp, fp, fn, tp_match = self.assign_kernel(out_pframe[i], tar_pframe[i])
            tpl.append(tp)
            fpl.append(fp)
            fnl.append(fn)
            tpml.append(tp_match)

        """Concat them back"""
        tp = emitter.EmitterSet.cat(tpl)
        fp = emitter.EmitterSet.cat(fpl)
        fn = emitter.EmitterSet.cat(fnl)
        tp_match = emitter.EmitterSet.cat(tpml)

        """Let tp and tp_match share the same id's. IDs of ground truth are copied to true positives."""
        if (tp_match.id == -1).all().item():
            tp_match.id = torch.arange(tp_match.__len__())
        tp.id = tp_match.id

        return self._return_match(tp=tp, fp=fp, fn=fn, tp_match=tp_match)


class NNMatching(MatcherABC):
    """
    A class to match outputs and targets based on 1neighbor nearest neighbor classifier.
    """

    def __init__(self, *, dist_lat=2.5, dist_ax=500, match_dims=3):
        """

        :param dist_lat: (float) lateral distance threshold
        :param dist_ax: (float) axial distance threshold
        :param match_dims: should we match the emitters only in 2D or also 3D
        """
        self.dist_lat_thresh = dist_lat
        self.dist_ax_thresh = dist_ax
        self.match_dims = match_dims

        self.nearest_neigh = NearestNeighbors(n_neighbors=1, metric='minkowski', p=2)

        if self.match_dims not in [2, 3]:
            raise ValueError("You must compare in either 2 or 3 dimensions.")

    @staticmethod
    def parse(param: dict):
        """

        :param param: parameter dict
        :return:
        """
        return NNMatching(**param['Evaluation'])

    def forward(self, output, target):
        """Forward arbitrary output and target set. Does not care about the frame_ix.

        :param output: (emitterset)
        :param target: (emitterset)
        :return tp, fp, fn, tp_match: (emitterset) true positives, false positives, false negatives, ground truth matched to the true pos
        """
        xyz_tar = target.xyz.numpy()
        xyz_out = output.xyz.numpy()

        if self.match_dims == 2:
            xyz_tar_ = xyz_tar[:, :2]
            xyz_out_ = xyz_out[:, :2]
        else:
            xyz_tar_ = xyz_tar
            xyz_out_ = xyz_out

        """If no emitter has been found, all are false negatives. No tp, no fp."""
        if xyz_out_.shape[0] == 0:
            tp = emitter.EmptyEmitterSet()
            fp = emitter.EmptyEmitterSet()
            fn = target
            tp_match = emitter.EmptyEmitterSet()

            return tp, fp, fn, tp_match

        if xyz_tar_.shape[0] != 0:
            nbrs = self.nearest_neigh.fit(xyz_tar_)

        else:
            """If there were no positives, no tp, no fn, all fp, no match."""
            tp = emitter.EmptyEmitterSet()
            fn = emitter.EmptyEmitterSet()
            fp = output
            tp_match = emitter.EmptyEmitterSet()

            return tp, fp, fn, tp_match

        distances_nn, indices = nbrs.kneighbors(xyz_out_)

        distances_nn = distances_nn.squeeze()
        indices = np.atleast_1d(indices.squeeze())

        xyz_match = target.get_subset(indices).xyz.numpy()

        # calculate distances lateral and axial seperately
        dist_lat = np.linalg.norm(xyz_out[:, :2] - xyz_match[:, :2], axis=1, ord=2)
        dist_ax = np.linalg.norm(xyz_out[:, [2]] - xyz_match[:, [2]], axis=1, ord=2)

        # remove those which are too far
        if self.match_dims == 3:
            is_tp = (dist_lat <= self.dist_lat_thresh) * (dist_ax <= self.dist_ax_thresh)
        elif self.match_dims == 2:
            is_tp = (dist_lat <= self.dist_lat_thresh)

        is_fp = ~is_tp

        indices[is_fp] = -1
        indices_cleared = indices[indices != -1]

        # create indices of targets
        tar_ix = np.arange(target.__len__())
        # remove indices which were found
        fn_ix = np.setdiff1d(tar_ix, indices)

        is_tp = torch.from_numpy(is_tp.astype(np.uint8)).type(torch.BoolTensor)
        is_fp = torch.from_numpy(is_fp.astype(np.uint8)).type(torch.BoolTensor)
        fn_ix = torch.from_numpy(fn_ix)

        tp = output.get_subset(is_tp)
        fp = output.get_subset(is_fp)
        fn = target.get_subset(fn_ix)
        tp_match = target.get_subset(indices_cleared)

        return self._return_match(tp=tp, fp=fp, fn=fn, tp_match=tp_match)
