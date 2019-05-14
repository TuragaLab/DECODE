import torch
from unittest import TestCase
import pytest

from deepsmlm.generic.emitter import EmitterSet, RandomEmitterSet


class TestEmitterSet(TestCase):

    def setUp(self):
        self.em2d = EmitterSet(xyz=torch.rand((25, 2)),
                               phot=torch.rand(25),
                               frame_ix=torch.zeros(25))

        frames = torch.arange(25)
        frames[[0, 1, 2]] = 1
        self.em3d = EmitterSet(xyz=torch.rand((25, 3)),
                               phot=torch.rand(25),
                               frame_ix=frames)

    def test_init(self):
        # 2D input get's converted to 3D with zeros
        self.assertEqual(3, self.em2d.xyz.shape[1], "2D converted to 3D")
        self.assertEqual(3, self.em3d.xyz.shape[1], "3D")

    def test_split_in_frames(self):
        splits = self.em2d.split_in_frames(None, None)
        self.assertEqual(1, splits.__len__(), "Only one frame ix.")

        splits = self.em3d.split_in_frames(None, None)
        self.assertEqual(self.em3d.frame_ix.max() - self.em3d.frame_ix.min() + 1,
                         splits.__len__(),
                         "Frame ix with wholes.")

        """Test negative numbers in Frame ix."""
        neg_frames = EmitterSet(torch.rand((3, 3)),
                                torch.rand(3),
                                torch.tensor([-1, 0., 1]))
        splits = neg_frames.split_in_frames(None, None)
        self.assertEqual(3, splits.__len__())
        splits = neg_frames.split_in_frames(0, None)
        self.assertEqual(2, splits.__len__())

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
