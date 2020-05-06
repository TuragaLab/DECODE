import pytest
import torch

import deepsmlm.simulation.psf_kernel as psf_kernel
from deepsmlm.generic import EmitterSet, CoordinateOnlyEmitter, RandomEmitterSet, EmptyEmitterSet, test_utils as tutil
from deepsmlm.neuralfitter import target_generator


class TestTargetGenerator:

    @pytest.fixture()
    def targ(self):
        """
        Setup dummy target generator for inheritors.

        """

        class DummyTarget(target_generator.TargetGenerator):
            def __init__(self, xextent, yextent, img_shape):
                super().__init__(ix_low=0, ix_high=0)
                self.xextent = xextent
                self.yextent = yextent
                self.img_shape = img_shape

                self.delta = psf_kernel.DeltaPSF(xextent=self.xextent,
                                                 yextent=self.yextent,
                                                 img_shape=self.img_shape)

            def forward_(self, xyz: torch.Tensor, phot: torch.Tensor, frame_ix: torch.Tensor, ix_low, ix_high):
                return self.delta.forward(xyz, phot, None, ix_low, ix_high).unsqueeze(1)

        xextent = (-0.5, 63.5)
        yextent = (-0.5, 63.5)
        img_shape = (64, 64)
        return DummyTarget(xextent, yextent, img_shape)

    @pytest.fixture(scope='class')
    def fem(self):
        return EmitterSet(xyz=torch.tensor([[0., 0., 0.]]), phot=torch.Tensor([1.]), frame_ix=torch.tensor([0]),
                          xy_unit='px')

    def test_shape(self, targ, fem):
        """
        Tests the frame_shape of the target output

        Args:
            targ:
            fem:

        """

        out = targ.forward(fem)

        """Tests"""
        assert out.dim() == 4, "Wrong dimensionality."
        assert out.size()[-2:] == torch.Size(targ.img_shape), "Wrong output shape."

    @pytest.mark.parametrize("ix_low,ix_high", [(0, 0), (-1, 1)])
    @pytest.mark.parametrize("em_data", [EmptyEmitterSet(xy_unit='px'), RandomEmitterSet(10, xy_unit='px')])
    def test_default_range(self, targ, ix_low, ix_high, em_data):
        targ.ix_low = ix_low
        targ.ix_high = ix_high

        """Run"""
        out = targ.forward(em_data)

        """Assertions"""
        assert out.size(0) == ix_high - ix_low + 1


class TestUnifiedEmbeddingTarget(TestTargetGenerator):

    @pytest.fixture()
    def targ(self):
        xextent = (-0.5, 63.5)
        yextent = (-0.5, 63.5)
        img_shape = (64, 64)

        return target_generator.UnifiedEmbeddingTarget(xextent, yextent, img_shape, roi_size=5, ix_low=0, ix_high=5)

    @pytest.fixture()
    def random_emitter(self):
        em = RandomEmitterSet(1000)
        em.frame_ix = torch.randint_like(em.frame_ix, low=-20, high=30)

        return em

    def test_eq_centralpx_delta(self, targ, random_emitter):
        """Check whether central pixels agree with delta function"""

        """Run"""
        mask, ix = targ._get_central_px(random_emitter.xyz, random_emitter.frame_ix)
        mask_delta = targ._delta_psf._fov_filter.clean_emitter(random_emitter.xyz)
        ix_delta = targ._delta_psf.px_search(random_emitter.xyz[mask], random_emitter.frame_ix[mask])

        """Assert"""
        assert (mask == mask_delta).all()
        for ix_el, ix_el_delta in zip(ix, ix_delta):
            assert (ix_el == ix_el_delta).all()

    @pytest.mark.parametrize("roi_size", torch.tensor([1, 3, 5, 7]))
    def test_roi_px(self, targ, random_emitter, roi_size):
        """Setup"""
        targ.__init__(xextent=targ.xextent, yextent=targ.yextent, img_shape=targ.img_shape,
                      roi_size=roi_size, ix_low=targ.ix_low, ix_high=targ.ix_high)

        """Run"""
        mask, ix = targ._get_central_px(random_emitter.xyz, random_emitter.frame_ix)
        batch_ix, x_ix, y_ix, off_x, off_y, id = targ._get_roi_px(*ix)

        """Assert"""
        assert (batch_ix.unique() == ix[0].unique()).all()
        assert (x_ix >= 0).all()
        assert (y_ix >= 0).all()
        assert (x_ix <= 63).all()
        assert (y_ix <= 63).all()
        assert batch_ix.size() == off_x.size()
        assert off_x.size() == off_y.size()

        expct_vals = torch.arange(-(targ._roi_size - 1) // 2, (targ._roi_size - 1) // 2 + 1)

        assert (off_x.unique() == expct_vals).all()
        assert (off_y.unique() == expct_vals).all()

    def test_forward(self, targ):
        """Test a couple of handcrafted cases"""

        # one emitter outside fov the other one inside
        em_set = CoordinateOnlyEmitter(torch.tensor([[-50., 0., 0.], [15.1, 19.6, 250.]]), xy_unit='px')
        em_set.phot = torch.tensor([5., 4.])

        out = targ.forward(em_set)[0]  # single frame
        assert tutil.tens_almeq(out[:, 15, 20], torch.tensor([1., 4., 0.1, -0.4, 250.]), 1e-5)
        assert tutil.tens_almeq(out[:, 16, 20], torch.tensor([0., 4., -0.9, -0.4, 250.]), 1e-5)
        assert tutil.tens_almeq(out[:, 15, 21], torch.tensor([0., 4., 0.1, -1.4, 250.]), 1e-5)
