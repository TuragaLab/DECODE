import torch

from decode import emitter
from decode import simulation
from decode import neuralfitter


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
    tar_em_filter = emitter.process.EmitterFilterGeneric(phot=lambda p: p > 100)

    tar = neuralfitter.utils.processing.TransformSequence(
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

    em = emitter.factory(10, xy_unit="px")
    frames = tar.forward(em)
