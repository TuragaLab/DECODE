import torch
import pytest

from unittest.mock import patch, Mock

from decode.utils import hardware


@pytest.mark.parametrize("device_str,device,ix", [('cpu', 'cpu', None),
                                                  ('cuda', 'cuda', None),
                                                  ('cuda:1', 'cuda', 1),
                                                  ('cud:1', 'err', None)])
def test__specific_device_by_str(device_str, device, ix):

    if device != 'err':
        device_out, ix_out = hardware._specific_device_by_str(device_str)
        assert device_out == device
        assert ix_out == ix

    else:
        with pytest.raises(ValueError):
            hardware._specific_device_by_str(device_str)


@pytest.mark.parametrize("device_cap, device_cap_str", [((3, 5), "3.5"), ((7, 5), "7.5")])
def test_get_device_capability(device_cap, device_cap_str):

    with patch('torch.cuda.get_device_capability', return_value=device_cap):
        if type(device_cap_str) != type(hardware.get_device_capability()) or len(device_cap_str) != len(hardware.get_device_capability()):
            raise NotImplementedError("Sanity check of mock failed.")

    with patch('torch.cuda.get_device_capability', return_value=device_cap):
        assert hardware.get_device_capability() == device_cap_str


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Only works with CUDA.")
@pytest.mark.parametrize("size_low,size_high,expct", [(1, 5000, None),
                                                      (5000, 35000, 'err')])
def test_get_max_batch_size(size_low, size_high, expct):
    def dummy(x):
        return x ** 2 + 2 * x + x.sqrt()

    x_size = (512, 512)

    if expct is None:
        bs = hardware.get_max_batch_size(dummy, x_size, 'cuda:0', size_low, size_high)
        dummy(torch.rand(bs, *x_size, device='cuda:0'))  # check if it actually works

    else:
        with pytest.raises(RuntimeError) as err:
            hardware.get_max_batch_size(dummy, x_size, 'cuda:0', size_low, size_high)
        assert "Lowest possible batch size is outside of provided bounds." == str(err.value)
