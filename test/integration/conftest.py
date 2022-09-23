import pytest
from pathlib import Path
from omegaconf import OmegaConf, DictConfig

from decode import io


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
def cfg_trainable(cfg) -> DictConfig:
    # ToDo: A trainable cfg, i.e. all necessary assets and directories present

    cfg.Simulation.intensity.mean = 5000
    cfg.Simulation.intensity.std = 1000

    cfg.Simulation.lifetime_avg = 1.

    return cfg
