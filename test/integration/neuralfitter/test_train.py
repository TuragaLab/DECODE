import inspect

import pytest
import torch.nn
from omegaconf import OmegaConf

from decode.generic import asset_handler
from decode import simulation
from decode import neuralfitter
from decode.neuralfitter.train import train


@pytest.fixture
def cfg(repo_dir):
    p = repo_dir / "config/config.yaml"
    cfg = OmegaConf.load(p)

    # overwrite hardware, because testing is only on cpu
    cfg.Hardware.device = "cpu"
    cfg.Hardware.device_simulation = "cpu"

    return cfg


@pytest.mark.parametrize("no_op", [True, False])
def test_setup_logger(no_op, cfg, tmpdir):
    cfg.Logging.no_op = no_op
    for logger in cfg.Logging.logger.values():
        logger.offline = True
        logger.save_dir = str(tmpdir)

    l = train.setup_logger(cfg)
    l[0].log_metrics({"a": 5})


@pytest.fixture
def path_bead_cal(scope="file"):
    return asset_handler.load_asset("bead_cal")


def test_setup_psf(path_bead_cal, cfg):
    cfg.InOut.calibration_file = path_bead_cal

    psf = train.setup_psf(cfg)
    assert isinstance(psf, simulation.psf_kernel.CubicSplinePSF)


def test_setup_background(cfg):
    bg = train.setup_background(cfg)
    assert isinstance(bg, simulation.background.Background)


@pytest.mark.parametrize("preset", ["Perfect", None])
def test_setup_noise(preset, cfg):
    cfg.CameraPreset = preset

    noise = train.setup_noise(cfg)
    assert isinstance(noise, simulation.camera.Camera)
    if preset == "Perfect":
        assert isinstance(noise, simulation.camera.PerfectCamera)
    else:
        assert isinstance(noise, simulation.camera.Photon2Camera)


@pytest.mark.parametrize("fn", [
    train.setup_structure,
    train.setup_model,
    train.setup_loss,
    train.setup_em_filter,
    train.setup_tar_disable,
    train.setup_post_process_frame_emitter,
    train.setup_matcher,
])
def test_setup_atomic_signature(fn, cfg):
    # this tests that the annotated return type is actually returned
    # only possible for `atomic` setup, i.e. functions with no other dependency
    # than the bare cfg file
    sig = inspect.signature(fn)
    out = fn(cfg)
    assert isinstance(out, sig.return_annotation)


def test_setup_optimizer(cfg):
    model = train.setup_model(cfg)
    o = train.setup_optimizer(model, cfg)

    assert isinstance(o, torch.optim.Optimizer)


def test_setup_scheduler(cfg):
    model = train.setup_model(cfg)
    opt = train.setup_optimizer(model, cfg)
    sched = train.setup_scheduler(opt, cfg)

    assert isinstance(sched, torch.optim.lr_scheduler.StepLR)


def test_setup_tar(cfg):
    train.setup_tar(None, None, cfg)


def test_tar_tensor_parameter(cfg):
    tar = train.setup_tar_tensor_parameter(None, None, cfg)
    assert isinstance(tar, neuralfitter.target_generator.TargetGenerator)


def test_setup_post_process(cfg):
    train.setup_post_process(cfg)
