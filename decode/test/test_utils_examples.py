from decode.utils import examples


def test_load_gateway():

    """Run"""
    gate = examples.load_gateway()

    """Assertions"""
    assert isinstance(gate, dict)
    assert gate['code'] == 'https://github.com/TuragaLab/DECODE/archive/master.zip'
