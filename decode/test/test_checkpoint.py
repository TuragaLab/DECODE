from pathlib import Path
import pytest

from ..utils import checkpoint


class TestCheckpoint:

    @pytest.fixture()
    def ckpt(self):
        path = Path('dummy_ckpt.pt')
        yield checkpoint.CheckPoint(path)
        path.unlink()

    def test_save_load(self, ckpt):
        ckpt.dump('a', 'b', 'c', 42, 'l')

        ckpt_re = checkpoint.CheckPoint.load(ckpt.path)
        assert ckpt.__dict__ == ckpt_re.__dict__
