import os
import pytest
import torch

import deepsmlm.neuralfitter.arguments as param
import deepsmlm.generic.inout.write_load_param as wlp

"""Root folder"""
deepsmlm_root = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)),
                 os.pardir, os.pardir)) + '/'


def test_write_param():
    filename_in = deepsmlm_root + 'deepsmlm/test/assets/test_param_for_write.json'
    filename_out = deepsmlm_root + 'deepsmlm/test/assets/dummy.json'
    param = wlp.load_params(filename_in)
    wlp.write_params(filename_out, param)
    exists = os.path.isfile(filename_out)
    assert exists
    os.remove(filename_out)


def test_load_params():
    filename = deepsmlm_root + 'deepsmlm/test/assets/test_param_for_load.json'

    wlp.load_params(filename)
    assert True
