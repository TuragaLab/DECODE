import torch

from decode import emitter
from decode import simulation
from decode import neuralfitter
from decode.neuralfitter import target_generator


def test_simulation():
    """
    Tests combination of structure sampler, psf, bg, microscope and samples from it
    """
    # constants
    psf = simulation.psf_kernel.DeltaPSF((-0.5, 31.5), (-0.5, 31.5), (32, 32))
    noise = simulation.noise.Poisson()
    m = simulation.microscope.Microscope(psf=psf, noise=noise, frame_range=(-5, 5))

    # samplers
    struct = simulation.structures.RandomStructure(
        (-0.5, 31.5), (-0.5, 31.5), (-500.0, 500.0)
    )
    color = simulation.code.Code([0, 1])
    em_sampler = simulation.sampler.EmitterSamplerBlinking(
        structure=struct,
        intensity=(1000.0, 100.0),
        em_num=50.0,
        lifetime=2.0,
        frame_range=(0, 100),
        code=color,
        xy_unit="px",
    )
    bg = simulation.background.UniformBackground((1.0, 100.0), size=(32, 32))

    # sample
    em = em_sampler.sample()
    bg_sample = bg.sample()
    frames = m.forward(em, bg_sample)

    assert frames.size() == torch.Size([10, 32, 32])


def test_target():
    from unittest import mock

    lane_emitter = target_generator.TargetGeneratorChain(
        [
            neuralfitter.target_generator.ParameterListTarget(
                n_max=100,
                xextent=(-0.5, 31.5),
                yextent=(-0.5, 31.5),
                ix_low=-5,
                ix_high=5,
                squeeze_batch_dim=False,
            ),
            neuralfitter.scale_transform.ParameterListRescale(
                phot_max=1000.0,
                z_max=1000.0,
                bg_max=100,
            ),
        ]
    )
    lane_bg = mock.MagicMock()
    lane_bg.forward = lambda x, y, _0, _1: y

    tar = target_generator.TargetGeneratorFork(
        components=[lane_emitter._components[0], lane_bg],
        merger=None,
    )

    em = emitter.factory(frame_ix=[-6, -5, 10], phot=torch.rand(3) * 1000, xy_unit="px")
    bg = torch.rand(10, 64, 64)

    (tar_em, tar_em_mask), tar_bg = tar.forward(em, bg)
