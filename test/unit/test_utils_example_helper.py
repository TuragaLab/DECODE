from decode.utils import example_helper


def test_load_gateway():

    """Run"""
    gate = example_helper.load_gateway()

    """Assertions"""
    assert isinstance(gate, dict)
    assert gate['code'] == 'https://github.com/TuragaLab/DECODE/archive/master.zip'
