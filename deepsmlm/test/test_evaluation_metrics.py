import unittest
from unittest import TestCase

from deepsmlm.evaluation.match_emittersets import NNMatching
from deepsmlm.generic.emitter import EmitterSet
from deepsmlm.evaluation.metric_library import *


class TestExpandedPairwiseDistances(TestCase):

    def setUp(self):
        pass

    def test_expanded_pairwise_distances(self):
        x = torch.tensor([[0., 0., 0.]])
        y = torch.tensor([[0., 0., 0.], [1., 1., 1.]])

        dist_mat = expanded_pairwise_distances(x, y)
        self.assertTrue((dist_mat == torch.tensor([[0., sqrt(3)]])).all())


class TestNNMatching(TestCase):

    def setUp(self):
        self.dist_lat = 2.5
        self.dist_ax = 500.
        self.test_object = NNMatching(self.dist_lat, self.dist_ax)

    def test_simple_pair_0(self):
        out = EmitterSet(torch.tensor([[0., 0., 0.]]),
                         phot=torch.tensor([1.]),
                         frame_ix=torch.tensor([0]))

        target = EmitterSet(torch.tensor([[2.4, 0., 500.]]),
                            phot=torch.tensor([1.]),
                            frame_ix=torch.tensor([0]))

        tp, fp, fn, _ = self.test_object.forward(out, target)
        self.assertEqual(tp.num_emitter, 1)
        self.assertEqual(fp.num_emitter, 0)
        self.assertEqual(fn.num_emitter, 0)

    def test_simple_pair_1(self):
        out = EmitterSet(torch.tensor([[0., 0., 0.]]),
                         phot=torch.tensor([1.]),
                         frame_ix=torch.tensor([0]))

        target = EmitterSet(torch.tensor([[2.4, 2.4, 0.]]),
                            phot=torch.tensor([1.]),
                            frame_ix=torch.tensor([0]))

        tp, fp, fn, _ = self.test_object.forward(out, target)
        self.assertEqual(tp.num_emitter, 0)
        self.assertEqual(fp.num_emitter, 1)
        self.assertEqual(fn.num_emitter, 1)

    def test_simple_pair_2(self):
        out = EmitterSet(torch.tensor([[0., 0., 0.]]),
                         phot=torch.tensor([1.]),
                         frame_ix=torch.tensor([0]))

        target = EmitterSet(torch.tensor([[0, 2.4, 501.]]),
                            phot=torch.tensor([1.]),
                            frame_ix=torch.tensor([0]))

        tp, fp, fn, _ = self.test_object.forward(out, target)
        self.assertEqual(tp.num_emitter, 0)
        self.assertEqual(fp.num_emitter, 1)
        self.assertEqual(fn.num_emitter, 1)


if __name__ == '__main__':
    unittest.main()
