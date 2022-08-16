from typing import Optional, Any
from unittest import mock

import numpy as np
import pytest
import torch

from decode.emitter import emitter
from decode.neuralfitter import target_generator


@pytest.fixture
def tar_pseudo_abstract():
    # such that we can still test the abstract class
    class _TarGen(target_generator.TargetGenerator):
        def forward(self, em: emitter.EmitterSet, aux: Optional[Any] = None) -> Any:
            raise NotImplementedError

    class _EmitterFilter:
        @staticmethod
        def forward(em):
            return em[em.phot >= 100.]

    class _Scaler:
        @staticmethod
        def forward(x):
            return x / 10

    return _TarGen(-5, 5, filter=_EmitterFilter(), scaler=_Scaler())


@pytest.mark.parametrize("tar", ["tar_pseudo_abstract"])
def test_targen_limit_shift_frames(tar, request):
    t = request.getfixturevalue(tar)
    em = emitter.factory(frame_ix=[-17, 23, -4, -5, 5])  # only one left should be -4 --> 1

    em_out = t._limit_shift_frames(em)
    assert em_out.frame_ix.tolist() == [1, 0]


@pytest.mark.parametrize("tar", ["tar_pseudo_abstract"])
def test_targen_filter_emitters(tar, request):
    tar = request.getfixturevalue(tar)

    em = emitter.factory(phot=[99, 100, 101])
    em_out = tar._filter_emitters(em)

    assert em != em_out
    assert (em_out.phot >= 100).all()


@pytest.mark.parametrize("tar", ["tar_pseudo_abstract"])
def test_targen_scaler(tar, request):
    tar = request.getfixturevalue(tar)

    x = torch.ones(10) * 10
    x_out = tar._scale(x)

    assert x_out is not x
    assert (x_out == 1.).all()


@pytest.mark.parametrize("switch", [True, False])
@pytest.mark.parametrize("ignore_ix", [True, False])
def test_target_gaussian_mixture(switch, ignore_ix):
    m_filter = mock.MagicMock()
    m_filter.forward.side_effect = lambda x: x
    m_scaler = mock.MagicMock()
    m_scaler.forward.side_effect = lambda x: x
    m_bg_lane = mock.MagicMock()
    m_bg_lane.forward.side_effect = lambda x: x + 5

    tar = target_generator.TargetGaussianMixture(
        filter=m_filter,
        scaler=m_scaler,
        switch=target_generator.DisableAttributes(-1) if switch else None,
        aux_lane=m_bg_lane,
        n_max=100,
        ix_low=-5,
        ix_high=5,
        ignore_ix=ignore_ix,
    )

    em = emitter.factory(phot=[10.], xyz=[[1., 2., 3.]], xy_unit="px")
    aux = torch.rand(10, 32, 32)

    (tar_em, tar_mask), aux_out = tar.forward(em, aux)
    m_filter.forward.assert_called_once()
    m_scaler.forward.assert_called_once()

    if switch:
        (tar_em[..., -1] == 0.).all()
    else:
        (tar_em[..., -1] == 3.).all()

    if ignore_ix:
        tar_em.size() == torch.Size([100, 4])
        tar_mask.size() == torch.Size([100])
    else:
        tar_em.size() == torch.Size([10, 100, 4])
        tar_mask.size() == torch.Size([10, 100])

    assert (aux_out > 5).all()


def _mock_tar_emitter_factory():
    """
    Produces a mock target generator that has the apropriate signature and outputs
    random frames of batch dim that equals the emitters frame span.
    """

    class _MockTargetGenerator:
        @staticmethod
        def forward(em, bg, ix_low, ix_high):
            n_frames = em.frame_ix.max() - em.frame_ix.min() + 1
            return torch.rand(n_frames, 32, 32)

    return _MockTargetGenerator()


def test_tar_chain():
    class _MockRescaler:
        @staticmethod
        def forward(x: torch.Tensor):
            return x / x.max()

    tar = target_generator.TargetGeneratorChain(
        [_mock_tar_emitter_factory(), _MockRescaler()]
    )

    out = tar.forward(emitter.factory(frame_ix=[-5, 5]), None)

    assert out.max() == 1.0


@pytest.mark.parametrize("merge", [None, torch.cat])
def test_tar_fork(merge):
    if merge is not None:
        merge = target_generator.TargetGeneratorMerger(fn=merge)

    tar = target_generator.TargetGeneratorFork(
        [_mock_tar_emitter_factory(), _mock_tar_emitter_factory()],
        merger=merge,
    )

    out = tar.forward(emitter.factory(frame_ix=[-5, 5]))

    if merge is None:
        assert len(out) == 2
        assert out[0].size() == torch.Size([11, 32, 32])
        assert out[1].size() == torch.Size([11, 32, 32])
    else:
        assert out.size() == torch.Size([22, 32, 32])


def test_tar_merge():
    tar = target_generator.TargetGeneratorMerger(fn=lambda x, y: torch.cat([x, y]))
    out = tar.forward(torch.rand(5, 32, 32), torch.rand(5, 32, 32))

    assert out.size() == torch.Size([10, 32, 32])


@pytest.mark.parametrize("attr", [[], ["em"], ["bg"], ["em", "bg"]])
def test_tar_forwarder(attr):
    tar = target_generator.TargetGeneratorForwarder(attr)

    em = mock.MagicMock()
    bg = mock.MagicMock()

    out = tar.forward(em, bg)

    if len(attr) == 0:
        assert out is None
    elif len(attr) != 1:
        assert len(out) == len(attr)
    else:
        if "em" in attr:
            assert out is em
        if "bg" in attr:
            assert out is bg


@pytest.mark.parametrize("ignore_ix", [True, False])
def test_paramlist(ignore_ix):
    tar = target_generator.ParameterList(
        n_max=100,
        xy_unit="px",
        ix_low=0,
        ix_high=3,
        ignore_ix=ignore_ix,
    )

    em = emitter.EmitterSet(
        xyz=[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
        phot=[3.0, 2.0],
        frame_ix=[0, 2],
        xy_unit="px",
    )

    tar, mask = tar.forward(em)

    assert mask.dtype == torch.bool
    assert mask.sum() == len(em)

    if not ignore_ix:
        assert tar.size() == torch.Size([3, 100, 4])
        assert mask.size() == torch.Size([3, 100])

        # manual emitter checks
        np.testing.assert_array_equal(tar[0, 0, 0], em[0].phot)
        np.testing.assert_array_equal(tar[0, 0, 1:], em[0].xyz.squeeze())
        np.testing.assert_array_equal(tar[2, 0, 0], em[1].phot)
        np.testing.assert_array_equal(tar[2, 0, 1:], em[1].xyz.squeeze())

        # check that everything but the filled out emitters are nan
        assert torch.isnan(tar[0, 1:]).all()
        assert torch.isnan(tar[1]).all()
        assert torch.isnan(tar[2, 1:]).all()

    else:
        assert tar.size() == torch.Size([100, 4])
        assert mask.size() == torch.Size([100])

        np.testing.assert_array_equal(tar[:2, 0], em.phot)
        np.testing.assert_array_equal(tar[:2, 1:], em.xyz)
        assert torch.isnan(tar[2:]).all()


def test_disable_attr():
    t = target_generator.DisableAttributes(1, 5.)

    x = torch.rand(1, 3, 7)
    out = t.forward(x)

    assert out is not x
    assert (out[..., 1] == 5.).all()
