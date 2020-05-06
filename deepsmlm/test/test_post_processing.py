import pytest
import torch

from deepsmlm.generic import emitter, test_utils
from deepsmlm.neuralfitter import post_processing


class TestPostProcessingAbstract:

    @pytest.fixture(params=["batch-set", "frame-set"], scope="function")
    def post(self, request):
        class PostProcessingMock(post_processing.PostProcessing):
            def forward(self):
                return emitter.EmptyEmitterSet()

        return PostProcessingMock(xy_unit=None, px_size=None, return_format=request.param)

    @pytest.mark.parametrize("return_format", [None, 'batch_set', 'frame_set', 'emitters'])
    def test_sanity(self, post, return_format):
        """
        Tests the sanity checks

        """
        with pytest.raises(ValueError):
            post.return_format = return_format
            post.sanity_check()

    def test_filter(self, post):

        assert not post.skip_if(torch.rand((1, 3, 32, 32)))


class TestNoPostProcessing(TestPostProcessingAbstract):

    @pytest.fixture()
    def post(self):
        return post_processing.NoPostProcessing()

    def test_forward(self, post):

        out = post.forward(torch.rand((256, 2, 64, 64)))
        assert isinstance(out, emitter.EmptyEmitterSet)


class TestConsistentPostProcessing(TestPostProcessingAbstract):

    @pytest.fixture()
    def post(self):
        return post_processing.ConsistencyPostprocessing(raw_th=0.1, em_th=0.5, xy_unit='px', img_shape=(32, 32),
                                                         ax_th=None, lat_th=0.5, match_dims=2,
                                                         return_format='batch-set')

    @pytest.mark.parametrize("return_format", [None, 'batch_set', 'frame_set', 'emitters'])
    def test_excpt(self, post, return_format):
        """
        Tests the sanity checks and forward expected exceptions

        """
        with pytest.raises(ValueError):
            post.__init__(raw_th=0.1, em_th=0.6, xy_unit='px', img_shape=(32, 32), lat_th=0.,
                          return_format=return_format)

        with pytest.raises(ValueError):
            post.forward(torch.rand((1, 2, 32, 32)))

        with pytest.raises(ValueError):
            post.forward(torch.rand((1, 7, 32, 32)))

    @pytest.mark.xfail(condition=not torch.cuda.is_available(), reason="CUDA not available on this machine.")
    def test_forward_cuda(self, post):
        p = torch.zeros((2, 1, 32, 32)).cuda()
        out = torch.zeros((2, 5, 32, 32)).cuda()
        p[1, 0, 2, 4] = 0.6
        p[1, 0, 2, 6] = 0.6
        p[0, 0, 0, 0] = 0.3
        p[0, 0, 0, 1] = 0.4

        out[0, 2, 0, 0] = 0.3
        out[0, 2, 0, 1] = 0.5
        out[1, 2, 2, 4] = 1.
        out[1, 2, 2, 6] = 1.2

        _ = post.forward(torch.cat((p, out), 1))

    def test_multi_worker(self, post):

        """Setup"""
        p = torch.zeros((2, 1, 32, 32))
        out = torch.zeros((2, 5, 32, 32))
        p[1, 0, 2, 4] = 0.6
        p[1, 0, 2, 6] = 0.6
        p[0, 0, 0, 0] = 0.3
        p[0, 0, 0, 1] = 0.4

        out[0, 2, 0, 0] = 0.3
        out[0, 2, 0, 1] = 0.5
        out[1, 2, 2, 4] = 1.
        out[1, 2, 2, 6] = 1.2

        out[:, 4] = torch.rand_like(out[:, 4])

        """Run"""
        post.num_workers = 0
        em0 = post.forward(torch.cat((p, out), 1))

        post.num_workers = 4
        em1 = post.forward(torch.cat((p, out), 1))

        """Assert (equal outcome)"""
        for i in range(len(em0)):
            assert em0[i] == em1[i]

    def test_easy_case(self, post):
        """
        Easy case, i.e. isolated active pixels.

        Args:
            post: fixture

        """

        """Setup"""
        p = torch.zeros((2, 1, 32, 32))
        out = torch.zeros((2, 5, 32, 32))
        p[0, 0, 0, 0] = 0.3
        p[0, 0, 0, 2] = 0.4
        p[1, 0, 2, 4] = 0.6
        p[1, 0, 2, 6] = 0.6

        out[0, 2, 0, 0] = 0.3
        out[0, 2, 0, 2] = 0.5
        out[1, 2, 2, 4] = 1.
        out[1, 2, 2, 6] = 1.2

        """Run"""
        p_out, feat_out = post._forward_raw_impl(p, out)
        em_out = post.forward(torch.cat((p, out), 1))

        """Assertions"""
        assert test_utils.tens_almeq(p, p_out)
        assert test_utils.tens_almeq(out, feat_out)

        assert isinstance(em_out, emitter.EmitterSet)
        assert len(em_out) == 2
        assert (em_out.prob >= post.em_th).all()

    def test_hard_cases(self, post):
        """Non-isolated emitters."""

        """Setup"""
        p = torch.zeros((3, 1, 32, 32))
        out = torch.zeros((3, 5, 32, 32))
        p[0, 0, 0, 0] = 0.7  # isolated
        p[0, 0, 0, 2] = 0.7  # isolated
        p[1, 0, 2, 4] = 0.6  # should be merged
        p[1, 0, 2, 5] = 0.6  # should be merged
        p[2, 0, 4, 4] = 0.7  # should not be merged
        p[2, 0, 4, 5] = 0.7  # should not be merged

        out[1, 1, 2, 4] = 20.  # should be merged
        out[1, 1, 2, 5] = 20.2  # %
        out[2, 2, 4, 4] = 49.  # should not be merged
        out[2, 2, 4, 5] = 49.51  # %

        """Run"""
        em_out = post.forward(torch.cat((p, out), 1))

        """Assertions"""
        # First frame
        assert len(em_out.get_subset_frame(0, 0)) == 2
        assert (em_out.get_subset_frame(0, 0).prob == 0.7).all()

        # Second frame
        assert len(em_out.get_subset_frame(1, 1)) == 1
        assert (em_out.get_subset_frame(1, 1).prob.item() > 0.6)

        # Third frame
        assert len(em_out.get_subset_frame(2, 2)) == 2
        assert (em_out.get_subset_frame(2, 2).prob == 0.7).all()

    @pytest.mark.parametrize("x,expct", [(torch.ones((2, 6, 32, 32)), True),
                                         (torch.zeros((2, 6, 32, 32)), False),
                                         (torch.tensor([[0.5, 0., 0.], [0., 0., 0.]]).unsqueeze(0).unsqueeze(0), False)])
    def test_filter(self, post, x, expct):

        post.skip_th = 0.2
        assert post.skip_if(x) is expct


@pytest.mark.skip("Deprecated function.")
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

#
# @pytest.fixture(scope='module')
# def cc():
#     return post.ConnectedComponents(0.1, 2)
#
#
# @pytest.fixture(scope='module')
# def cc_offset():
#     return post.CC5ChModel(0.3, 0., 2)
#
# def test_crlbdist():
#     """
#     Tests the cramer rao lower bound distance function between x and y
#     :return:
#     """
#     """Check for zero tensors and equal tensors."""
#     X = torch.zeros((32, 3))
#     Y = torch.zeros_like(X)
#
#     XCrlb = torch.ones_like(X)
#     YCrlb = torch.ones_like(Y)
#
#     out = post.crlb_squared_distance(X, Y, XCrlb, YCrlb)
#     assert tutil.tens_almeq(out, torch.zeros_like(X[:, 0]))
#
#     X = torch.rand((32, 3))
#     Y = X
#     out = post.crlb_squared_distance(X, Y, XCrlb, YCrlb)
#     assert tutil.tens_almeq(out, torch.zeros_like(X[:, 0]))
#
#
# def test_connected_components(cc):
#     p_map = torch.tensor([[0., 0., 0.5], [0., 0., 0.5], [0., 0., 0.]])
#     clusix = torch.tensor([[0, 0, 1.], [0, 0, 1], [0, 0, 0]])
#     assert torch.eq(clusix, cc.compute_cix(p_map)).all()
#
#
# class TestCC5ChModel:
#     testdata = []
#
#     def test_average_features(self, cc_offset):
#         p_map = torch.tensor([[0., 0., 0.5], [0., 0., 0.5], [0., 0., 0.]])
#         clusix = torch.tensor([[0, 0, 1.], [0, 0, 1], [0, 0, 0]])
#         features = torch.cat((
#             p_map.unsqueeze(0),
#             torch.tensor([[0, 0, .5], [0, 0, 1], [0, 0, 0]]).unsqueeze(0),
#             torch.tensor([[0, 0, .5], [0, 0, 1], [0, 0, 0]]).unsqueeze(0)
#         ), 0)
#         out_feat, out_p = cc_offset.average_features(features, clusix, p_map)
#
#         expect_outcome_feat = torch.tensor([[0.5, 0.75, 0.75]])
#         expect_p = torch.tensor([1.])
#
#         assert torch.eq(expect_outcome_feat, out_feat).all()
#         assert torch.eq(expect_p, out_p).all()
#
#     def test_forward(self, cc_offset):
#         p_map = torch.tensor([[1., 0., 0.5], [0., 0., 0.5], [0., 0., 0.]])
#         clusix = torch.tensor([[1, 0, 2.], [0, 0, 2], [0, 0, 0]])
#         features = torch.cat((
#             p_map.unsqueeze(0),
#             torch.tensor([[100., 10000., 500.], [0, 0, 500.], [0, 0, 0]]).unsqueeze(0),
#             torch.tensor([[0, 0, .5], [0, 0, 1], [0, 0, 0]]).unsqueeze(0),
#             torch.tensor([[0, 0, .5], [0, 0, 1], [0, 0, 0]]).unsqueeze(0),
#             torch.tensor([[0, 0, -500.], [0, 0, -250], [0, 0, 0]]).unsqueeze(0),
#         ), 0)
#         features.unsqueeze_(0)
#         em_list = cc_offset.forward(features)
#         assert em_list.__len__() == 1
