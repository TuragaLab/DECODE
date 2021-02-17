from pathlib import Path

import pytest

import decode.utils.param_io as wlp
from decode.utils import types
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


def test_load_reference_param():
    """
    Depends on reference.yaml in utils/references_files

    """
    param = wlp.load_reference()

    assert isinstance(param, dict)
    assert param['CameraPreset'] is None
    assert param['Evaluation']['dist_ax'] == 500.0


def test_load_by_reference_param():
    """
    Check that param that misses values is filled as the reference file is.
    Depends on  utils/references_files

    """

    """Run"""
    param_file = test_dir / Path('assets/param.yaml')
    with asset_handler.RMAfterTest(param_file):

        wlp.save_params(param_file, {'X': 1})
        param = wlp.load_params(param_file)

    """Assertions"""
    assert param.Hardware.device_simulation == 'cuda:0'


@pytest.mark.parametrize("mode_missing", ['exclude', 'include'])
def test_autofill_dict(mode_missing):
    """Setup"""
    a = {'a': 1,
         'z': {'x': 4},
         'only_in_a': 2,
         }

    ref = {'a': 2,
           'b': None,
           'c': 3,
           'z': {'x': 5, 'y': 6},
           }

    """Run"""
    a_ = wlp.autofill_dict(a, ref, mode_missing=mode_missing)

    """Assert"""
    assert a_['a'] == 1
    assert a_['b'] is None
    assert a_['c'] == 3
    assert a_['z']['x'] == 4
    assert a_['z']['y'] == 6
    if mode_missing == 'exclude':
        assert 'only_in_a' not in a_.keys()
    elif mode_missing == 'include':
        assert a_['only_in_a'] == 2


def test_write_param():
    filename = test_dir / Path('assets/test_param_for_load.json')
    asset_handler.AssetHandler().auto_load(filename)
    param = wlp.ParamHandling().load_params(filename)

    filename_out = test_dir / Path('assets/dummy.yml')
    with asset_handler.RMAfterTest(filename):
        wlp.ParamHandling().write_params(filename_out, param)
        assert isinstance(param, types.RecursiveNamespace)


def test_set_autoscale_param():
    param = types.RecursiveNamespace()
    param.Simulation = types.RecursiveNamespace()
    param.Scaling = types.RecursiveNamespace()
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


def test_copy_reference_param():

    path = test_dir / Path(f'assets/refs')
    path.mkdir(exist_ok=True)

    with asset_handler.RMAfterTest(path, recursive=True):
        wlp.copy_reference_param(path)

        assert (path / 'reference.yaml').exists()
        assert (path / 'param_friendly.yaml').exists()
