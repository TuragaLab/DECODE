import pytest
from pathlib import Path
from omegaconf import OmegaConf, DictConfig

from decode import io, utils
from decode.generic import asset_handler


@pytest.fixture
def null_hash():
    # sha 256 hash of empty file
    return "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"


@pytest.fixture
def repo_dir() -> Path:
    # ToDo: Not stable when packaged
    return Path(__file__).parents[2]


@pytest.fixture
def secrets(repo_dir):
    p = repo_dir / "config/secrets.yaml"
    if not p.exists():
        raise FileNotFoundError("Secret file 'secrets.yaml' not existing.")

    return OmegaConf.load(p)


@pytest.fixture
def cfg() -> DictConfig:
    cfg = io.param.load_reference()

    # overwrite hardware, because testing is only on cpu
    cfg.Hardware.device = "cpu"
    cfg.Hardware.device_simulation = "cpu"

    return cfg


@pytest.fixture
def cfg_trainable(cfg, tmpdir) -> DictConfig:
    """
    A trainable config. Successively fill out all parameters necessary for training.

    Args:
        cfg: reference config
    """
    cfg = utils.types.RecursiveNamespace(**OmegaConf.to_object(cfg))

    cfg.Simulation.bg_uniform = (10., 100.)
    cfg.Simulation.intensity.mean = 5000
    cfg.Simulation.intensity.std = 1000
    cfg.Simulation.lifetime_avg = 1.

    # Todo: Replace scaling by auto
    cfg.Scaling.input_scale = 100.
    cfg.Scaling.input_offset = 10.
    cfg.Scaling.bg_max = 120.
    cfg.Scaling.phot_max = 13000.
    cfg.Scaling.z_max = 900.

    cfg.Paths.calibration = asset_handler.load_asset("bead_cal")
    cfg.Paths.experiment = tmpdir / "model"
    cfg.Paths.logging = tmpdir / "log"

    cfg.Trainer.max_epochs = 3

    return cfg
