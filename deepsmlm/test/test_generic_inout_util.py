import pytest
from pathlib import Path

from deepsmlm.generic.inout import util


deepsmlm_root = Path(__file__).parent.parent
dummy_path = Path('a_dummy_path.txt')


@pytest.mark.parametrize("root", [deepsmlm_root, str(deepsmlm_root)])
@pytest.mark.parametrize("path", [dummy_path, str(dummy_path)])
def test_add_root_relative(root, path):
    assert util.add_root_relative(path, root) == deepsmlm_root / dummy_path
