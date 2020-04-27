import warnings
from abc import ABC, abstractmethod
from collections import namedtuple
from functools import partial

import numpy as np
import torch
from deprecated import deprecated
from sklearn.neighbors import NearestNeighbors

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

    @classmethod
    def parse(cls, param):
        return cls(match_dims=param.Evaluation.match_dims,
                   dist_lat=param.Evaluation.dist_lat,
                   dist_ax=param.Evaluation.dist_ax,
                   dist_vol=param.Evaluation.dist_vol)

    def filter(self, xyz_out, xyz_tar) -> torch.Tensor:
        """
        Filter kernel to rule out unwanted matches. Batch implemented, i.e. input can be 2 or 3 dimensional, where the
        latter dimensions are the dimensions of interest.

        Args:
            xyz_out: output coordinates, shape :math: `(B x) N x 3`
            xyz_tar: target coordinates, shape :math: `(B x) M x 3`

        Returns:
            filter_mask (torch.Tensor): boolean of size (B x) N x M
        """
        if xyz_out.dim() == 3:
            assert xyz_out.size(0) == xyz_tar.size(0)
            sque_ret = False  # no squeeze before return
        else:
            xyz_out = xyz_out.unsqueeze(0)
            xyz_tar = xyz_tar.unsqueeze(0)
            sque_ret = True  # squeeze before return

        filter_mask = torch.ones((xyz_out.size(0), xyz_out.size(1), xyz_tar.size(1))).bool()  # dim: B x N x M

        if self.dist_lat is not None:
            dist_mat = torch.cdist(xyz_out[:, :, :2], xyz_tar[:, :, :2], p=2)
            filter_mask[dist_mat > self.dist_lat ** 2] = 0

        if self.dist_ax is not None:
            dist_mat = torch.cdist(xyz_out[:, :, [2]], xyz_tar[:, :, [2]], p=2)
            filter_mask[dist_mat > self.dist_ax ** 2] = 0

        if self.dist_vol is not None:
            dist_mat = torch.cdist(xyz_out, xyz_tar, p=2)
            filter_mask[dist_mat > self.dist_vol ** 2] = 0

        if sque_ret:
            filter_mask = filter_mask.squeeze(0)

        return filter_mask

    @staticmethod
    def _rule_out_kernel(dists):
        """
        Kernel which goes through the distance matrix, picks shortest distance and assign match.
        Actual 'greedy' kernel

        Args:
            dists: distance matrix

        Returns:

        """
        assert dists.dim() == 2

        if dists.numel() == 0:
            return torch.zeros((0,)).long(), torch.zeros((0,)).long()

        dists_ = dists.clone()

        match_list = []
        while not (dists_ == float('inf')).all():
            ix = np.unravel_index(dists_.argmin(), dists_.shape)
            dists_[ix[0]] = float('inf')
            dists_[:, ix[1]] = float('inf')

            match_list.append(ix)

        if match_list.__len__() >= 1:
            assignment = torch.tensor(match_list).long()
        else:
            assignment = torch.zeros((0, 2)).long()

        return assignment[:, 0], assignment[:, 1]

    def _match_kernel(self, xyz_out, xyz_tar, filter_mask):
        """

        Args:
            xyz_out: N x 3  - no batch implementation currently
            xyz_tar: M x 3 - no batch implementation currently
            filter_mask: N x M - not batched

        Returns:
            tp_ix_: (boolean) index for xyz_out
            tp_match_ix_: (boolean) index for matching xyz_tar

        """
        assert filter_mask.dim() == 2
        assert filter_mask.size() == torch.Size([xyz_out.size(0), xyz_tar.size(0)])

        if self.match_dims == 2:
            dist_mat = torch.cdist(xyz_out[None, :, :2], xyz_tar[None, :, :2], p=2).squeeze(0)
        elif self.match_dims == 3:
            dist_mat = torch.cdist(xyz_out[None, :, :], xyz_tar[None, :, :], p=2).squeeze(0)
        else:
            raise ValueError

        dist_mat[~filter_mask] = float('inf')  # rule out matches by filter
        tp_ix, tp_match_ix = self._rule_out_kernel(dist_mat)

        tp_ix_bool = torch.zeros(xyz_out.size(0)).bool()
        tp_ix_bool[tp_ix] = 1
        tp_match_ix_bool = torch.zeros(xyz_tar.size(0)).bool()
        tp_match_ix_bool[tp_match_ix] = 1

        return tp_ix, tp_match_ix, tp_ix_bool, tp_match_ix_bool

    def forward(self, output, target):
        """

        Args:
            output:
            target:

        Returns:

        """
        """Setup split in frames. Determine the frame range automatically so as to cover everything."""
        if len(output) >= 1 and len(target) >= 1:
            frame_low = output.frame_ix.min() if output.frame_ix.min() < target.frame_ix.min() else target.frame_ix.min()
            frame_high = output.frame_ix.max() if output.frame_ix.max() > target.frame_ix.max() else target.frame_ix.max().item()
        elif len(output) >= 1:
            frame_low = output.frame_ix.min()
            frame_high = output.frame_ix.max()
        elif len(target) >= 1:
            frame_low = target.frame_ix.min()
            frame_high = target.frame_ix.max()
        else:
            return (emitter.EmptyEmitterSet(xy_unit=target.xyz, px_size=target.px_size), ) * 4

        out_pframe = output.split_in_frames(frame_low, frame_high)
        tar_pframe = target.split_in_frames(frame_low, frame_high)

        tpl, fpl, fnl, tpml = [], [], [], []  # true positive list, false positive list, false neg. ...

        """Assign the emitters framewise"""
        for out_f, tar_f in zip(out_pframe, tar_pframe):
            filter_mask = self.filter(out_f.xyz_nm, tar_f.xyz_nm)  # batch implemented
            tp_ix, tp_match_ix, tp_ix_bool, tp_match_ix_bool = self._match_kernel(out_f.xyz_nm, tar_f.xyz_nm, filter_mask)  # non batch impl.

            tpl.append(out_f[tp_ix])
            tpml.append(tar_f[tp_match_ix])
            fpl.append(out_f[~tp_ix_bool])
            fnl.append(tar_f[~tp_match_ix_bool])

        """Concat them back"""
        tp = emitter.EmitterSet.cat(tpl)
        fp = emitter.EmitterSet.cat(fpl)
        fn = emitter.EmitterSet.cat(fnl)
        tp_match = emitter.EmitterSet.cat(tpml)

        """Let tp and tp_match share the same id's. IDs of ground truth are copied to true positives."""
        if (tp_match.id == -1).all().item():
            tp_match.id = torch.arange(len(tp_match)).type(tp_match.id.dtype)

        tp.id = tp_match.id.type(tp.id.dtype)

        return self._return_match(tp=tp, fp=fp, fn=fn, tp_match=tp_match)


@deprecated
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
