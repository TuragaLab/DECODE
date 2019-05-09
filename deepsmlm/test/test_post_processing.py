import torch
from unittest import TestCase

import deepsmlm.neuralfitter.post_processing as post


class TestConnectedComponents(TestCase):
    def setUp(self) -> None:
        self.cc = post.ConnectedComponents(0.3, 0., 2)

    def test_compute_cix(self):
        p_map = torch.tensor([[0., 0., 0.5], [0., 0., 0.5], [0., 0., 0.]])
        clusix = torch.tensor([[0, 0, 1.], [0, 0, 1], [0, 0, 0]])
        assert torch.eq(clusix, self.cc.compute_cix(p_map)).all()


class TestConnectedComponentsOffset(TestCase):
    def setUp(self) -> None:
        self.cc = post.ConnectedComponentsOffset(0.3, 0., 2)

    def test_averagefeatures(self):
        p_map = torch.tensor([[0., 0., 0.5], [0., 0., 0.5], [0., 0., 0.]])
        clusix = torch.tensor([[0, 0, 1.], [0, 0, 1], [0, 0, 0]])
        features = torch.cat((
            p_map.unsqueeze(0),
            torch.tensor([[0, 0, .5], [0, 0, 1], [0, 0, 0]]).unsqueeze(0),
            torch.tensor([[0, 0, .5], [0, 0, 1], [0, 0, 0]]).unsqueeze(0)
        ), 0)
        out_feat, out_p = self.cc.average_features(features, clusix, p_map)

        expect_outcome_feat = torch.tensor([[0.5, 0.75, 0.75]])
        expect_p = torch.tensor([1.])

        assert torch.eq(expect_outcome_feat, out_feat).all()
        assert torch.eq(expect_p, out_p).all()

