import pytest
import dotmap
from collections import namedtuple
from deepsmlm.utils import dotmap

from pathlib import Path

import deepsmlm.utils.param_io as wlp

from . import asset_handler

"""Root folder"""
test_dir = str(Path(__file__).resolve().parent)


def test_load_params():

    filename = test_dir / Path('assets/test_param_for_load.json')
    asset_handler.AssetHandler().auto_load(filename)
    _ = wlp.ParamHandling().load_params(filename)

    with pytest.raises(FileNotFoundError):
        filename = test_dir / Path('assets/test_param_for_load2.json')
        _ = wlp.ParamHandling().load_params(filename)


def test_write_param():

    filename = test_dir / Path('assets/test_param_for_load.json')
    asset_handler.AssetHandler().auto_load(filename)
    param = wlp.ParamHandling().load_params(filename)

    filename_out = test_dir / Path('assets/dummy.yml')
    with asset_handler.RMAfterTest(filename):
        wlp.ParamHandling().write_params(filename_out, param)
        assert isinstance(param, dotmap.DotMap)


def test_set_autoscale_param():

    param = dotmap.DotMap({'ScalingMode': 'auto'})
    param.Simulation.intensity_mu_sig = (100., 1.)
    param.Simulation.bg_uniform = 10.
    param.Simulation.emitter_extent = (None, None, (-800., 800.))
    param.Scaling.input_scale = None
    param.Scaling.input_offset = None
    param.Scaling.phot_max = None
    param.Scaling.bg_max = None
    param.Scaling.z_max = None

    param = wlp.autoset_scaling(param)

    assert param.Scaling.input_scale == 2.
    assert param.Scaling.input_offset == 10.
    assert param.Scaling.bg_max == 12.
    assert param.Scaling.phot_max == 108.
    assert param.Scaling.z_max == 960.
