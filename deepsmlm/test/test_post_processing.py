import torch
import pytest

import deepsmlm.neuralfitter.post_processing as post
import deepsmlm.test.utils_ci as tutil


@pytest.fixture(scope='module')
def cc():
    return post.ConnectedComponents(0.1, 2)


@pytest.fixture(scope='module')
def cc_offset():
    return post.CC5ChModel(0.3, 0., 2)

def test_crlbdist():
    """
    Tests the cramer rao lower bound distance function between x and y
    :return:
    """
    """Check for zero tensors and equal tensors."""
    X = torch.zeros((32, 3))
    Y = torch.zeros_like(X)

    XCrlb = torch.ones_like(X)
    YCrlb = torch.ones_like(Y)

    out = post.crlb_squared_distance(X, Y, XCrlb, YCrlb)
    assert tutil.tens_almeq(out, torch.zeros_like(X[:, 0]))

    X = torch.rand((32, 3))
    Y = X
    out = post.crlb_squared_distance(X, Y, XCrlb, YCrlb)
    assert tutil.tens_almeq(out, torch.zeros_like(X[:, 0]))


def test_connected_components(cc):
    p_map = torch.tensor([[0., 0., 0.5], [0., 0., 0.5], [0., 0., 0.]])
    clusix = torch.tensor([[0, 0, 1.], [0, 0, 1], [0, 0, 0]])
    assert torch.eq(clusix, cc.compute_cix(p_map)).all()


class TestCC5ChModel:
    testdata = []

    def test_average_features(self, cc_offset):
        p_map = torch.tensor([[0., 0., 0.5], [0., 0., 0.5], [0., 0., 0.]])
        clusix = torch.tensor([[0, 0, 1.], [0, 0, 1], [0, 0, 0]])
        features = torch.cat((
            p_map.unsqueeze(0),
            torch.tensor([[0, 0, .5], [0, 0, 1], [0, 0, 0]]).unsqueeze(0),
            torch.tensor([[0, 0, .5], [0, 0, 1], [0, 0, 0]]).unsqueeze(0)
        ), 0)
        out_feat, out_p = cc_offset.average_features(features, clusix, p_map)

        expect_outcome_feat = torch.tensor([[0.5, 0.75, 0.75]])
        expect_p = torch.tensor([1.])

        assert torch.eq(expect_outcome_feat, out_feat).all()
        assert torch.eq(expect_p, out_p).all()

    def test_forward(self, cc_offset):
        p_map = torch.tensor([[1., 0., 0.5], [0., 0., 0.5], [0., 0., 0.]])
        clusix = torch.tensor([[1, 0, 2.], [0, 0, 2], [0, 0, 0]])
        features = torch.cat((
            p_map.unsqueeze(0),
            torch.tensor([[100., 10000., 500.], [0, 0, 500.], [0, 0, 0]]).unsqueeze(0),
            torch.tensor([[0, 0, .5], [0, 0, 1], [0, 0, 0]]).unsqueeze(0),
            torch.tensor([[0, 0, .5], [0, 0, 1], [0, 0, 0]]).unsqueeze(0),
            torch.tensor([[0, 0, -500.], [0, 0, -250], [0, 0, 0]]).unsqueeze(0),
        ), 0)
        features.unsqueeze_(0)
        em_list = cc_offset.forward(features)
        assert em_list.__len__() == 1


class TestSpeiser:

    @pytest.fixture(scope='class')
    def speis(self):
        return post.SpeiserPost(0.3, 0.6, 'frames')

    @pytest.fixture(scope='class')
    def feat(self):
        feat = torch.zeros((2, 5, 32, 32))

        feat[0, 0, 5, 5] = .4 + 1e-7
        feat[0, 0, 5, 6] = .4
        feat[0, 1, 5, 5] = 10.
        feat[0, 1, 5, 6] = 20.

        return feat

    def test_run(self, speis, feat):

        output = speis.forward(feat)
        assert torch.eq(torch.tensor(feat.size()), torch.tensor(output.size())).all()
        assert torch.eq(torch.tensor([15., 0.]), output[0, 1, 5, 5:7]).all()

    def test_trace(self, speis, feat):
        x = torch.rand((2, 5, 64, 64))
        traced_script_module = torch.jit.trace(post.speis_post_functional, x)

        """Test whether the results are the same."""
        output_speis = speis.forward(feat)
        output_trace = traced_script_module(feat)
        assert torch.eq(output_speis, output_trace).all()

        x = torch.rand(32, 5, 64, 64)
        assert torch.eq(speis.forward(x), traced_script_module(x)).all()


class TestConsistentPostProcessing:
    @pytest.fixture(scope='class')
    def post(self):
        return post.ConsistencyPostprocessing(0.1, final_th=0.5, out_format='emitters_framewise')

    def test_init_sanity_check(self):
        post.ConsistencyPostprocessing(0.1, 0.6, lat_threshold=30, ax_threshold=200., vol_threshold=None, match_dims=3)

    def test_easy(self, post):
        p = torch.zeros((2, 1, 32, 32)).cuda()
        out = torch.zeros((2, 4, 32, 32)).cuda()
        p[1, 0, 2, 4] = 0.6
        p[1, 0, 2, 6] = 0.6
        p[0, 0, 0, 0] = 0.3
        p[0, 0, 0, 1] = 0.4

        out[0, 2, 0, 0] = 0.3
        out[0, 2, 0, 1] = 0.5
        out[1, 2, 2, 4] = 1.
        out[1, 2, 2, 6] = 1.2

        em = post.forward(torch.cat((p, out),1))
