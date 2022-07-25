import pytest
import torch

from decode import emitter
from decode import simulation
from decode import neuralfitter
from decode.neuralfitter import target_generator


@pytest.fixture
def samplers() -> tuple[
    simulation.sampler.EmitterSamplerBlinking, simulation.background.UniformBackground
]:
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
    bg = simulation.background.UniformBackground((1.0, 100.0), size=(10, 32, 32))

    return em_sampler, bg


@pytest.fixture
def microscope():
    psf = simulation.psf_kernel.DeltaPSF((-0.5, 31.5), (-0.5, 31.5), (32, 32))
    noise = simulation.noise.Poisson()
    return simulation.microscope.Microscope(psf=psf, noise=noise, frame_range=(-5, 5))


def test_simulation(samplers, microscope):
    """
    Tests combination of structure sampler, psf, bg, microscope and samples from it
    """
    em_sampler, bg_sampler = samplers

    # sample
    em = em_sampler.sample()
    bg_sample = bg_sampler.sample()
    frames = microscope.forward(em, bg_sample)

    assert frames.size() == torch.Size([10, 32, 32])


@pytest.fixture
def target():
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
    lane_bg = target_generator.TargetGeneratorForwarder(["bg"])

    return target_generator.TargetGeneratorFork(
        components=[lane_emitter._components[0], lane_bg],
        merger=None,
    )


def test_target(target):
    em = emitter.factory(frame_ix=[-6, -5, 10], phot=torch.rand(3) * 1000, xy_unit="px")
    bg = torch.rand(10, 64, 64)

    (tar_em, tar_em_mask), tar_bg = target.forward(em, bg)
    assert tar_bg is bg


@pytest.fixture
def post_model():
    return neuralfitter.scale_transform.InverseParamListRescale(
                phot_max=1000.0,
                z_max=1000.0,
                bg_max=100,
            )


@pytest.fixture
def post_processor() -> neuralfitter.utils.processing.TransformSequence:
    return neuralfitter.utils.processing.TransformSequence(
        (
            neuralfitter.coord_transform.Offset2Coordinate(
                xextent=(-0.5, 63.5),
                yextent=(-0.5, 63.5),
                img_shape=(64, 64),
            ),
            neuralfitter.post_processing.LookUpPostProcessing(
                raw_th=0.1,
                pphotxyzbg_mapping=[0, 1, 2, 3, 4, -1],
                photxyz_sigma_mapping=[5, 6, 7, 8],
                xy_unit="px",
                px_size=[110.0, 120.0],
            ),
        )
    )


def test_post_processor(post_processor):
    x = torch.rand(2, 10, 64, 64)
    _ = post_processor.forward(x)


@pytest.fixture
def processor(target, post_model, post_processor):
    frame_scale = neuralfitter.scale_transform.AmplitudeRescale(scale=1000.)
    em_filter = emitter.process.EmitterFilterGeneric(phot=lambda x: x > 100.0)

    return neuralfitter.process.ProcessingSupervised(
        pre_input=frame_scale,
        pre_tar=em_filter,
        tar=target,
        post_model=post_model,
        post=post_processor,
    )


def test_processor(samplers, microscope, processor):
    em_sampler, bg_sampler = samplers
    em = em_sampler.sample()
    bg = bg_sampler.sample()

    frames = microscope.forward(em, bg)

    _ = processor.input(frames)
    _ = processor.tar(em, bg)
    _ = processor.post(torch.rand(2, 10, 64, 64))


def test_sampler(samplers, processor, microscope):
    em_sampler, bg_sampler = samplers

    em = em_sampler.sample()
    bg = bg_sampler.sample()

    s = neuralfitter.sampler.SamplerSupervised(
        em=em,
        bg=bg,
        proc=processor,
        mic=microscope
    )
    s.sample()

    s.input[5]
    s.target[5]
