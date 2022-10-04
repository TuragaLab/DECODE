from pathlib import Path

import pytest

from decode.neuralfitter.train import train


@pytest.mark.slow
def test_train(cfg_trainable):
    exp = Path(cfg_trainable.Paths.experiment)
    log = Path(cfg_trainable.Paths.logging)

    train.train(cfg_trainable)

    assert exp.is_dir()
    assert log.is_dir()
