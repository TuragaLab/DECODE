import pytest
from omegaconf import OmegaConf

from decode.generic.asset_handler import auto_asset
from decode.simulation import psf_kernel
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


def test_setup_psf(cfg):
    _test_setup_psf_impl(cfg)


@auto_asset("bead_cal")
def _test_setup_psf_impl(cfg, path_bead_cal):
    # outsourced because otherwise decorator does not work in conjunction with pytest
    cfg.InOut.calibration_file = path_bead_cal
    cfg.Hardware.device_simulation = "cpu"

    psf = train.setup_psf(cfg)
    assert isinstance(psf, psf_kernel.CubicSplinePSF)
