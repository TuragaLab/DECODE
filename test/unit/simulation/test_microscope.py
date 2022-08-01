import pytest
import torch
import numpy as np

from unittest import mock

from decode import emitter_factory, EmitterSet
from decode.simulation import microscope
from decode.simulation import noise as noise_lib
from decode.simulation import psf_kernel


@pytest.fixture
def psf():
    return psf_kernel.DeltaPSF((-0.5, 31.5), (-0.5, 39.5), (32, 40))


@pytest.fixture
def noise_poisson():
    return noise_lib.Poisson()


@pytest.mark.parametrize("bg", [None, torch.ones(32, 40)])
@pytest.mark.parametrize("noise", [None, mock.MagicMock()])
def test_microscope(bg, psf, noise):
    if noise is not None:
        noise.forward.side_effect = lambda x: x * 1e5

    m = microscope.Microscope(psf, noise, frame_range=(0, 10))
    em = emitter_factory(phot=torch.ones(10) * 1e5, xy_unit="px")

    frames = m.forward(em, bg)

    if noise is None:
        assert frames.max() == pytest.approx(1e5, rel=0.01)
    else:
        assert frames.max() == pytest.approx(1e10, rel=0.01)


def test_microscope_multi_channel():
    psf = [
        psf_kernel.DeltaPSF((0.0, 32.0), (0.0, 32.0), (32, 32)),
        psf_kernel.DeltaPSF((0.0, 32.0), (0.0, 32.0), (32, 32)),
    ]
    noise = [noise_lib.ZeroNoise(), noise_lib.ZeroNoise()]

    m = microscope.MicroscopeMultiChannel(psf, noise, (-5, 5), (-2, 0))
    em = emitter_factory(3, frame_ix=[-5, 0, 5], code=[-5, -1, 1], xy_unit="px")

    frames = m.forward(em)

    assert frames.size() == torch.Size([10, 2, 32, 32])
    assert (frames[:5] == 0).all()
    assert (frames[6:] == 0).all()
    assert not (frames[5, 1] == 0).all()


def test_microscope_channel_modifier():
    def _modifier_factory(factor_xyz, factor_phot, on_code: int):
        def mod(em: EmitterSet) -> EmitterSet:
            em = em.clone()

            xyz = em.xyz
            xyz[em.code == on_code] = em.icode[on_code].xyz * factor_xyz
            phot = em.phot
            phot[em.code == on_code] = em.icode[on_code].phot * factor_phot
            em.code[:] = on_code
            return em

        return mod

    splitter = microscope.MicroscopeChannelModifier(
        [
            _modifier_factory(0.5, 0.5, 0),
            _modifier_factory(2.0, 2.0, 1),
            _modifier_factory(4.0, 4.0, 2),
        ]
    )

    em_out = splitter.forward(
        emitter_factory(xyz=[[8., 10, 12], [10, 14, 18]], code=[0, 1])
    )

    assert isinstance(em_out, EmitterSet)
    assert len(em_out) == 6
    np.testing.assert_array_equal(em_out.code, torch.LongTensor([0, 0, 1, 1, 2, 2]))
    np.testing.assert_array_almost_equal(
        em_out.xyz,
        torch.Tensor([[4., 5., 6.],
                      [10., 14., 18.],
                      [8., 10., 12.],
                      [20, 28, 36],
                      [8, 10, 12],
                      [10, 14, 18]])
    )


def test_emitter_composite_attribute_modifier():
    # no special treatment for EmitterSet as of now, only aliasing actual implementation
    pass


def test_channel_coordinate_trafo_matrix():
    m = microscope.CoordTrafoMatrix(torch.rand(3, 3))

    xyz = torch.rand(10, 3)
    xyz_out = m.forward(xyz)

    assert xyz_out.size() == xyz.size()


@pytest.mark.parametrize("t,color,expct", [
    ([[1., 0], [0., 1]], [0], [1., 0]),
    ([[1., 0], [0., 1]], [1], [0., 1.]),
    ([[0., 1.], [1., 0]], [0], [0., 1.]),
])
def test_multi_choric_splitter_static(t, color, expct):
    t = torch.Tensor(t)
    m = microscope.MultiChoricSplitter(t)

    phot = torch.Tensor([1.])
    color = torch.LongTensor(color)
    phot_expct = torch.Tensor(expct).unsqueeze(0)

    phot_out = m.forward(phot, color)
    np.testing.assert_array_equal(phot_out, phot_expct)


def test_multi_choric_splitter_sample_transmission():
    t = torch.Tensor([
        [0.7, 0.3],
        [0.2, 0.8],
    ])
    t_sig = torch.ones(2, 2)
    m = microscope.MultiChoricSplitter(t, t_sig)

    t_sampled = m.sample_transmission()
    assert t_sampled.size() == t.size()
    assert (t_sampled >= 0.).all(), "Transmission matrix should non-negative."
    assert (t_sampled.sum(1) == 1.).all(), \
        "Transmission matrix should be row-wise normalized."


def test_expand_by_index():
    x = torch.Tensor([1000., 2000.])
    ix = torch.LongTensor([1, 2])
    ix_max = 4

    x_expct = torch.Tensor([
        [0, 1000., 0, 0],
        [0, 0, 2000., 0]
    ])

    x_out = microscope.MultiChoricSplitter._expand_col_by_index(x, ix, ix_max)
    np.testing.assert_array_equal(x_out, x_expct)
