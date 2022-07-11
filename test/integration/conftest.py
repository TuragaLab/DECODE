import pytest
from pathlib import Path
from omegaconf import OmegaConf


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
