import os
import pytest
import dotmap

import deepsmlm.generic.inout.write_load_param as wlp

"""Root folder"""
deepsmlm_root = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)),
                 os.pardir, os.pardir)) + '/'


def test_load_params():
    filename = deepsmlm_root + 'deepsmlm/test/assets/test_param_for_load.json'
    param = wlp.ParamHandling().load_params(filename)
    assert True

    with pytest.raises(ValueError):
        filename = deepsmlm_root + 'deepsmlm/test/assets/test_param_for_load.jsaon'
        _ = wlp.ParamHandling().load_params(filename)

    with pytest.raises(FileNotFoundError):
        filename = deepsmlm_root + 'deepsmlm/test/assets/test_param_for_load2.json'
        _ = wlp.ParamHandling().load_params(filename)


def test_write_param():
    filename = deepsmlm_root + 'deepsmlm/test/assets/test_param_for_load.json'
    param = wlp.ParamHandling().load_params(filename)

    filename_out = deepsmlm_root + 'deepsmlm/test/assets/dummy.yml'
    wlp.ParamHandling().write_params(filename_out, param)
    assert isinstance(param, dotmap.DotMap)
