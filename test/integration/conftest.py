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
def cfg_trainable(cfg, tmpdir, request) -> DictConfig:
    """
    A trainable config. Complete set of parameters for training

    Args:
        cfg: reference config
    """
    auto = param_auto.AutoConfig(fill=False, fill_test=True)

    cfg["Camera"][0]["baseline"] = 100
    cfg["Camera"][0]["e_per_adu"] = 40
    cfg["Camera"][0]["em_gain"] = 100
    cfg["Camera"][0]["read_sigma"] = 80.
    cfg["Camera"][0]["spur_noise"] = 0.

    cfg["Scaling"]["input"]["aux"] = None

    cfg["Simulation"]["bg"] = {0: {"uniform": (10., 100.)}}
    cfg["Simulation"]["intensity"]["mean"] = 5000
    cfg["Simulation"]["intensity"]["std"] = 1000
    cfg["Simulation"]["lifetime_avg"] = 1.
    cfg["Simulation"]["code"] = [0]

    cfg["Paths"]["calibration"] = asset_handler.load_asset("bead_cal")
    cfg["Paths"]["experiment"] = str(tmpdir / "model")
    cfg["Paths"]["logging"] = str(tmpdir / "log")

    cfg["Trainer"]["max_epochs"] = 3
    cfg = auto.parse(cfg)
    return cfg


@pytest.fixture
def cfg_multi(cfg_trainable) -> DictConfig:
    auto = param_auto.AutoConfig(fill=False, fill_test=True, auto_scale=False)
    cfg = cfg_trainable

    code = [0, 1, 2]

    cfg["Simulation"]["code"] = code
    cfg["Camera"] = {i: cfg["Camera"][0] for i in code}
    cfg["Simulation"]["bg"] = {
        i: cfg["Simulation"]["bg"][0] for i in code
    }

    # need to overwrite test again because it changes
    cfg["Test"] = {"samples": cfg["Test"]["samples"]}
    cfg = auto.parse(cfg)

    return cfg
