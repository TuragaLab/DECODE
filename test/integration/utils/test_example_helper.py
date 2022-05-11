from decode.utils import example_helper


def test_load_gateway():
    gate = example_helper.load_gateway()

    assert isinstance(gate, dict)
    assert gate["code"] == "https://github.com/TuragaLab/DECODE/archive/master.zip"
