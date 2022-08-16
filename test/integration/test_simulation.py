import pytest
import torch

from decode import emitter
from decode import simulation
from decode import neuralfitter
from decode.neuralfitter import target_generator


@pytest.fixture
def samplers() -> tuple[
    simulation.sampler.EmitterSamplerBlinking, simulation.background.BackgroundUniform
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
    bg = simulation.background.BackgroundUniform((1.0, 100.0), size=(10, 32, 32))

    return em_sampler, bg


@pytest.fixture
def microscope():
    psf = simulation.psf_kernel.DeltaPSF((-0.5, 31.5), (-0.5, 31.5), (32, 32))
    noise = simulation.noise.Poisson()
    return simulation.microscope.Microscope(psf=psf, noise=noise, frame_range=(-5, 5))


@pytest.fixture
def target():
    scaler = neuralfitter.scale_transform.ScalerTargetList(
        phot=1000.0,
        z=1000.0,
    )
    filter = emitter.process.EmitterFilterFoV((-0.5, 31.5), (-0.5, 31.5))
    bg_lane = neuralfitter.scale_transform.ScalerAmplitude(100.0)
    return target_generator.TargetGaussianMixture(
        n_max=100,
        ix_low=None,
        ix_high=None,
        ignore_ix=True,
        scaler=scaler,
        filter=filter,
        aux_lane=bg_lane,
    )


def test_target(target):
    em = emitter.factory(frame_ix=[-6, -5, 10], phot=torch.rand(3) * 1000, xy_unit="px")
    bg = torch.rand(10, 64, 64)

    (tar_em, tar_em_mask), tar_bg = target.forward(em, bg)


@pytest.fixture
def post_model():
    return neuralfitter.scale_transform.ScalerModelOutput(
        phot=1000.0,
        z=1000.0,
        bg=100,
    )


@pytest.fixture
def post_processor():
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
    frame_scale = neuralfitter.scale_transform.ScalerAmplitude(scale=1000.0)
    em_filter = emitter.process.EmitterFilterGeneric(phot=lambda x: x > 100.0)

    return neuralfitter.process.ProcessingSupervised(
        pre_input=frame_scale,
        tar=target,
        post_model=post_model,
        post=post_processor,
    )


def test_processor(samplers, microscope, processor):
    em_sampler, bg_sampler = samplers
    em = em_sampler.sample()
    bg = bg_sampler.sample()

    frames = microscope.forward(em, bg)

    _ = processor.input(frames, em, bg)
    _ = processor.tar(em, bg)
    _ = processor.post(torch.rand(2, 10, 64, 64))


@pytest.mark.parametrize("num_workers", [0, 2])
def test_sampler_training(num_workers, samplers, microscope, target):
    # during training, we can not sample background once for all frames and simply add
    # it, as it needs to vary for different samples but stay constant within one window

    em_sampler, bg_sampler = samplers
    em = em_sampler.sample()
    bg = bg_sampler.sample(size=(10, 1, 32, 32))

    noise = microscope._noise
    microscope._noise = None

    # noise thing
    shared_input = neuralfitter.utils.process.InputMerger(noise)
    proc = neuralfitter.process.ProcessingSupervised(
        shared_input=shared_input,
        tar=target,
    )

    s = neuralfitter.sampler.SamplerSupervised(
        em=em,
        bg=bg,
        proc=proc,
        mic=microscope,
        bg_mode="sample",
        window=3,
    )
    s.sample()

    # wrap in a dataset and try dataloader
    ds = neuralfitter.dataset.DatasetGausianMixture(s.input, s.target)
    assert len(ds) == 10

    x, (tar_em, tar_mask, tar_bg) = ds[5]

    assert x.size() == torch.Size([3, 32, 32])
    assert tar_em.size() == torch.Size([100, 4])
    assert tar_mask.size() == torch.Size([100])
    assert tar_bg.size() == torch.Size([1, 32, 32])

    dl = torch.utils.data.DataLoader(ds, batch_size=4, num_workers=num_workers)
    assert len(dl) == 3

    x_batch, (tar_em_batch, tar_mask_batch, tar_bg_batch) = next(iter(dl))

    assert x_batch.size() == torch.Size([4, 3, 32, 32])
    assert tar_em_batch.size() == torch.Size([4, 100, 4])
    assert tar_mask_batch.size() == torch.Size([4, 100])
    assert tar_bg_batch.size() == torch.Size([4, 1, 32, 32])

    # # dummy model output
    # out = torch.rand(dl.batch_size, 10, 32, 32, requires_grad=True)
    #
    # loss = neuralfitter.loss.GaussianMMLoss(
    #     xextent=(-0.5, 31.5),
    #     yextent=(-0.5, 31.5),
    #     img_shape=(32, 32),
    #     device="cpu"
    # )
    # loss.forward(out, y, None)
