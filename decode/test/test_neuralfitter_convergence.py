from unittest import mock
import pytest

from decode.neuralfitter.utils import progress


class TestProgressCheck:

    @pytest.fixture()
    def checker(self):
        return progress.NoCheck()

    def test_call(self, checker):
        with mock.patch.object(checker, 'check_progress') as impl:
            checker()

        impl.assert_called_once()

    def test_check(self, checker):
        assert checker()


class TestHeuristicCheck(TestProgressCheck):

    @pytest.fixture()
    def checker(self):
        return progress.GMMHeuristicCheck(ref_epoch=1, emitter_avg=1.)

    @pytest.mark.parametrize("loss, epoch, converges", [
        ([1e8, 1e7, 1e6], (0, 1, 2), (True, False, False)),
        ([1e8, 99, 1], (0, 1, 2), (True, True, True)),
        ([1e8, 99, 1e8], (0, 1, 2), (True, True, True)),
    ])
    def test_check(self, checker, loss, epoch, converges):

        for e, l, c in zip(epoch, loss, converges):
            assert checker(l, e) == c
