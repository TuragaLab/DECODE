import inspect
from typing import get_args

import pytest
import torch.nn

from decode.generic import asset_handler
from decode import simulation
from decode.neuralfitter.train import setup_cfg


@pytest.mark.parametrize("no_op", [True, False])
def test_setup_logger(no_op, cfg, tmpdir):
    cfg.Paths.logging = str(tmpdir)
    cfg.Logging.no_op = no_op

    l = setup_cfg.setup_logger(cfg)

    l[0].log_metrics({"a": 5})


@pytest.fixture
def path_bead_cal(scope="file"):
    return asset_handler.load_asset("bead_cal")


def test_setup_psf(path_bead_cal, cfg):
    cfg.Paths.calibration = path_bead_cal

    psf = setup_cfg.setup_psf(cfg)
    assert isinstance(psf, simulation.psf_kernel.CubicSplinePSF)


def test_setup_background(cfg_trainable):
    bg, bg_val = setup_cfg.setup_background(cfg_trainable)
    assert isinstance(bg, simulation.background.Background)
    assert isinstance(bg_val, simulation.background.Background)


@pytest.mark.parametrize(
    "fn",
    [
        setup_cfg.setup_structure,
        setup_cfg.setup_code,
        setup_cfg.setup_trafo_coord,
        setup_cfg.setup_trafo_phot,
        setup_cfg.setup_model,
        setup_cfg.setup_loss,
        setup_cfg.setup_em_filter,
        setup_cfg.setup_frame_scaling,
        setup_cfg.setup_aux_scaling,
        setup_cfg.setup_tar_scaling,
        setup_cfg.setup_bg_scaling,
        setup_cfg.setup_post_model_scaling,
        setup_cfg.setup_post_process_frame_emitter,
        setup_cfg.setup_post_process_offset,
        setup_cfg.setup_post_process,
        setup_cfg.setup_matcher,
        setup_cfg.setup_evaluator,
        setup_cfg.setup_tar,
    ],
)
def test_setup_atomic_signature(fn, cfg):
    # this tests that the annotated return type is actually returned
    # only possible for `atomic` setup, i.e. functions with no other dependency
    # than the bare cfg file
    sig = inspect.signature(fn)
    out = fn(cfg)
    assert isinstance(out, sig.return_annotation)


@pytest.mark.parametrize(
    "fn",
    [
        setup_cfg.setup_cameras,
    ],
)
def test_setup_atomic_signature_sequences(fn, cfg):
    sig = inspect.signature(fn)
    out = fn(cfg)
    assert isinstance(out[0], get_args(sig.return_annotation)[0])


def test_setup_optimizer(cfg):
    model = setup_cfg.setup_model(cfg)
    o = setup_cfg.setup_optimizer(model, cfg)

    assert isinstance(o, torch.optim.Optimizer)


def test_setup_sampler(cfg_trainable):
    cfg = cfg_trainable
    n_train = cfg["Simulation"]["samples"]
    n_val = cfg["Test"]["samples"]

    s_train, s_val = setup_cfg.setup_sampler(cfg)

    for s, n in zip((s_train, s_val), (n_train, n_val)):
        s.sample()

        assert s.emitter.frame_ix.min() == 0  # could fail for very low densities
        assert s.emitter.frame_ix.max() == n - 1
        assert s.frame.size() == torch.Size([n, *cfg["Simulation"]["img_size"]])
        assert s.bg.size() == torch.Size([n, *cfg["Simulation"]["img_size"]])
        assert s.input[:].size() == torch.Size(
            [n, cfg["Trainer"]["frame_window"], *cfg["Simulation"]["img_size"]]
        )
