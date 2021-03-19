import pytest
from pathlib import Path

import decode.utils.param_io

decode_root = Path(__file__).parent.parent
dummy_path = Path('a_dummy_path.txt')


@pytest.mark.parametrize("root", [decode_root, str(decode_root)])
@pytest.mark.parametrize("path", [dummy_path, str(dummy_path)])
def test_add_root_relative(root, path):
    assert decode.utils.param_io.add_root_relative(path, root) == decode_root / dummy_path
