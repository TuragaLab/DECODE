import torch
from unittest import TestCase

import deepsmlm.neuralfitter.post_processing as post


class TestConnectedComponents(TestCase):
    def setUp(self) -> None:
        self.cc = post.ConnectedComponents(0.3, 0., 2)

    def test_compute_cix(self):
        p_map = torch.tensor([[0., 0., 0.5], [0., 0., 0.5], [0., 0., 0.]])
        clusix = torch.tensor([[0, 0, 1.], [0, 0, 1], [0, 0, 0]])
        assert True == torch.eq(clusix, self.cc.compute_cix(p_map))