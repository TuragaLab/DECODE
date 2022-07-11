import pytest
from omegaconf import OmegaConf

from decode.generic import asset_handler
from decode import simulation
from decode.neuralfitter.train import train


@pytest.fixture
def cfg(repo_dir):
    p = repo_dir / "config/config.yaml"
    return OmegaConf.load(p)


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
    cfg.Hardware.device_simulation = "cpu"

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


def test_setup_prior(cfg):
    p = train.setup_prior(cfg)
    assert isinstance(p, simulation.structures.StructurePrior)
