import os
import pytest
import dotmap

from pathlib import Path

import deepsmlm.generic.inout.write_load_param as wlp

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
