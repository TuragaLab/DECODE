import pytest

from decode.utils import param_auto


@pytest.mark.parametrize("mode_missing", ["exclude", "include"])
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


def test_auto_config_fill():
    pass
