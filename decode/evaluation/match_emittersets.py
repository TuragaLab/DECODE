import warnings
from abc import ABC, abstractmethod
from collections import namedtuple

import numpy as np
import torch

from decode.generic import emitter as emitter


class EmitterMatcher(ABC):
    """
    Abstract emitter matcher class.

    """

    _return_match = namedtuple('MatchResult', ['tp', 'fp', 'fn', 'tp_match'])  # return-type as namedtuple

    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, output: emitter.EmitterSet, target: emitter.EmitterSet) -> _return_match:
        """
        All implementations shall implement this forward method which takes output and reference set of emitters
        and outputs true positives, false positives, false negatives and matching ground truth (matched to the true positives).

        Args:
            output: output set of emitters
            target: reference set of emitters

        Returns:
            (emitter.EmitterSet, emitter.EmitterSet, emitter.EmitterSet, emitter.EmitterSet)

                - **tp**: true positives
                - **fp**: false positives
                - **fn**: false negatives
                - **tp_match**: ground truths that have been matched to the true positives

        """
        raise NotImplementedError


class GreedyHungarianMatching(EmitterMatcher):
    """
    Matching emitters in a greedy 'hungarian' fashion, by using best first search.

    """

    def __init__(self, *, match_dims: int, dist_ax: float = None, dist_lat: float = None, dist_vol: float = None):
        """

        Args:
            match_dims: match in 2D or 3D
            dist_lat: lateral tolerance radius
            dist_ax: axial tolerance threshold
            dist_vol: volumetric tolerance radius
        """
        super().__init__()

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
            filter_mask[dist_mat > self.dist_lat] = 0

        if self.dist_ax is not None:
            dist_mat = torch.cdist(xyz_out[:, :, [2]], xyz_tar[:, :, [2]], p=2)
            filter_mask[dist_mat > self.dist_ax] = 0

        if self.dist_vol is not None:
            dist_mat = torch.cdist(xyz_out, xyz_tar, p=2)
            filter_mask[dist_mat > self.dist_vol] = 0

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

    def forward(self, output: emitter.EmitterSet, target: emitter.EmitterSet):

        """Setup split in frames. Determine the frame range automatically so as to cover everything."""
        if len(output) >= 1 and len(target) >= 1:
            frame_low = output.frame_ix.min() if output.frame_ix.min() < target.frame_ix.min() else target.frame_ix.min()
            frame_high = output.frame_ix.max() if output.frame_ix.max() > target.frame_ix.max() else target.frame_ix.max()
        elif len(output) >= 1:
            frame_low = output.frame_ix.min()
            frame_high = output.frame_ix.max()
        elif len(target) >= 1:
            frame_low = target.frame_ix.min()
            frame_high = target.frame_ix.max()
        else:
            return (emitter.EmptyEmitterSet(xy_unit=target.xyz, px_size=target.px_size),) * 4

        out_pframe = output.split_in_frames(frame_low.item(), frame_high.item())
        tar_pframe = target.split_in_frames(frame_low.item(), frame_high.item())

        tpl, fpl, fnl, tpml = [], [], [], []  # true positive list, false positive list, false neg. ...

        """Match the emitters framewise"""
        for out_f, tar_f in zip(out_pframe, tar_pframe):
            filter_mask = self.filter(out_f.xyz_nm, tar_f.xyz_nm)  # batch implemented
            tp_ix, tp_match_ix, tp_ix_bool, tp_match_ix_bool = self._match_kernel(out_f.xyz_nm, tar_f.xyz_nm,
                                                                                  filter_mask)  # non batch impl.

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
