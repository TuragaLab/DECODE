import pytest
from pathlib import Path

import torch.cuda
from omegaconf import OmegaConf, DictConfig

from decode import io
from decode.generic import asset_handler
from decode.utils import param_auto


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

    # overwrite hardware if cuda is not available
    if not torch.cuda.is_available():
        cfg["Hardware"]["device"]["training"] = "cpu"
        cfg["Hardware"]["device"]["simulation"] = "cpu"

    return cfg


@pytest.fixture
def cfg_trainable(cfg, tmpdir) -> DictConfig:
    """
    A trainable config. Complete set of parameters for training

    Args:
        cfg: reference config
    """
    auto = param_auto.AutoConfig(fill=False, fill_test=True)

    cfg.Simulation.bg[0].uniform = (10., 100.)
    cfg.Simulation.intensity.mean = 5000
    cfg.Simulation.intensity.std = 1000
    cfg.Simulation.lifetime_avg = 1.

    cfg.Paths.calibration = asset_handler.load_asset("bead_cal")
    cfg.Paths.experiment = str(tmpdir / "model")
    cfg.Paths.logging = str(tmpdir / "log")

    cfg.Trainer.max_epochs = 3

    cfg = auto.parse(cfg)

    return cfg
