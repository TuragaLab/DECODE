import pytest

from decode.utils import param_auto


@pytest.mark.parametrize("mode_missing", ["exclude", "include"])  # raise has sep. test
def test_autofill_dict(mode_missing):
    a = {
        "a": 1,
        "z": {"x": 4},
        "only_in_a": 2,
    }

    ref = {
        "a": 2,
        "b": None,
        "c": 3,
        "z": {"x": 5, "y": 6},
    }

    a_ = param_auto._autofill_dict(a, ref, mode_missing=mode_missing)

    assert a_["a"] == 1
    assert a_["b"] is None
    assert a_["c"] == 3
    assert a_["z"]["x"] == 4
    assert a_["z"]["y"] == 6
    if mode_missing == "exclude":
        assert "only_in_a" not in a_.keys()
    elif mode_missing == "include":
        assert a_["only_in_a"] == 2


def test_autofill_dict_raise():
    with pytest.raises(ValueError):
        param_auto._autofill_dict({"a": 2}, {"b": 42}, mode_missing="raise")


def test_auto_config_fill():
    auto = param_auto.AutoConfig()
    cfg_out = auto._fill(dict())
    assert cfg_out == param_auto.param.load_reference()


def test_auto_config_fill_test():
    auto = param_auto.AutoConfig()
    cfg = auto._fill(dict())  # get reference
    cfg["Simulation"]["samples"] = 1000
    cfg["Test"]["samples"] = 5
    cfg_out = auto._fill_test(cfg)

    assert cfg_out["Test"].keys() == cfg_out["Simulation"].keys()
    assert cfg_out["Test"]["samples"] == 5


def test_auto_config_fill_scaling():
    auto = param_auto.AutoConfig()
    cfg = auto._fill(dict())  # get reference

    # set necessary params
    cfg["Simulation"]["intensity"]["mean"] = 5000
    cfg["Simulation"]["intensity"]["std"] = 1000
    cfg["Simulation"]["bg"][0]["uniform"] = (80, 120)

    cfg_out = auto._auto_scale(cfg)
    cfg_scale = cfg_out["Scaling"]
    assert all ([v is not None for v in cfg_scale["input"]["frame"].values()])

