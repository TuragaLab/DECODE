import torch
from unittest import TestCase
import pytest

from deepsmlm.generic.emitter import EmitterSet, RandomEmitterSet, EmptyEmitterSet


class TestEmitterSet:

    @pytest.fixture(scope='class')
    def em2d(self):
        return EmitterSet(xyz=torch.rand((25, 2)),
                               phot=torch.rand(25),
                               frame_ix=torch.zeros(25))

    @pytest.fixture(scope='class')
    def em3d(self):
        frames = torch.arange(25)
        frames[[0, 1, 2]] = 1
        return EmitterSet(xyz=torch.rand((25, 3)),
                               phot=torch.rand(25),
                               frame_ix=frames)

    def test_init(self, em2d, em3d):
        # 2D input get's converted to 3D with zeros
        assert em2d.xyz.shape[1] == 3
        assert em3d.xyz.shape[1] == 3

    def test_split_in_frames(self, em2d, em3d):
        splits = em2d.split_in_frames(None, None)
        assert splits.__len__() == 1

        splits = em3d.split_in_frames(None, None)
        assert em3d.frame_ix.max() - em3d.frame_ix.min() + 1 == splits.__len__()

        """Test negative numbers in Frame ix."""
        neg_frames = EmitterSet(torch.rand((3, 3)),
                                torch.rand(3),
                                torch.tensor([-1, 0., 1]))
        splits = neg_frames.split_in_frames(None, None)
        assert  splits.__len__() == 3
        splits = neg_frames.split_in_frames(0, None)
        assert splits.__len__() == 2

    def test_adjacent_frame_split(self):
        xyz = torch.rand((500, 3))
        phot = torch.rand_like(xyz[:, 0])
        frame_ix = torch.randint_like(xyz[:, 0], low=-1, high=2)
        em = EmitterSet(xyz, phot, frame_ix)

        em_split = em.split_in_frames(-1, 1)
        assert (em_split[0].frame_ix == -1).all()
        assert (em_split[1].frame_ix == 0).all()
        assert (em_split[2].frame_ix == 1).all()

        em_split = em.split_in_frames(0, 0)
        assert em_split.__len__() == 1
        assert (em_split[0].frame_ix == 0).all()

        em_split = em.split_in_frames(-1, -1)
        assert em_split.__len__() == 1
        assert (em_split[0].frame_ix == -1).all()

        em_split = em.split_in_frames(1, 1)
        assert em_split.__len__() == 1
        assert (em_split[0].frame_ix == 1).all()

    def test_cat_emittersets(self):
        """
        Test the concatenation of two emittersets.
        :return:
        """
        sets = [RandomEmitterSet(50), RandomEmitterSet(20)]
        cat_sets = EmitterSet.cat_emittersets(sets, None, 1)
        assert 70 == cat_sets.num_emitter
        assert 0 == cat_sets.frame_ix[0]
        assert 1 == cat_sets.frame_ix[50]

        sets = [RandomEmitterSet(50), RandomEmitterSet(20)]
        cat_sets = EmitterSet.cat_emittersets(sets, [5, 50], None)
        assert 70 == cat_sets.num_emitter
        assert 5 == cat_sets.frame_ix[0]
        assert 50 == cat_sets.frame_ix[50]


def test_empty_emitterset():
    em = EmptyEmitterSet()
    assert 0 == em.num_emitter

