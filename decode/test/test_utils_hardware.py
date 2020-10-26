import pytest
from unittest.mock import patch, Mock

from decode.utils import hardware


@pytest.mark.parametrize("device_cap, device_cap_str", [((3, 5), "3.5"), ((7, 5), "7.5")])
def test_get_device_capability(device_cap, device_cap_str):

    if type(device_cap_str) != type(hardware.get_device_capability()) or len(device_cap_str) != len(hardware.get_device_capability()):
        raise NotImplementedError("Sanity check of mock failed.")

    with patch('torch.cuda.get_device_capability', return_value=device_cap):
        assert hardware.get_device_capability() == device_cap_str
