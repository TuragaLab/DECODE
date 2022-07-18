from unittest import mock

import pytest

from decode.neuralfitter import dataset


@pytest.mark.parametrize(
    "ix,window,pad,ix_expct",
    [
        (0, 1, None, 0),
        (0, 3, None, 1),
        (0, 5, None, 2),
        (0, 1, "same", 0),
        (0, 3, "same", 0),
    ],
)
def test_pad_index(ix, window, pad, ix_expct):
    s = dataset.IxShifter(mode=pad, window=window)
    assert s(ix) == ix_expct


@pytest.mark.parametrize("window,pad,n,len_expct",[
    (1, None, 10, 10),
    (3, None, 10, 8),
    (3, "same", 10, 10),
])
def test_pad_index_len(window, pad, n, len_expct):
    s = dataset.IxShifter(pad, window, n)
    assert len(s) == len_expct


class TestDatasetSMLM:
    @pytest.fixture
    def ds(self):
        return dataset.DatasetSMLM(
            sampler=mock.MagicMock(),
            frame_window=3,
            ix_mod=mock.MagicMock()
        )

    def test_len(self, ds):
        ds._ix_mod.__len__.return_value = 42
        assert len(ds) == 42

    def test_len_raw(self, ds):
        ds._sampler.__len__.return_value = 100
        assert ds._len_raw == 100

    def test_getitem(self, ds):
        sample = ds[42]

        assert len(sample) == 2
        ds._sampler.input.__getitem__.assert_called_once_with(42)
        ds._sampler.target.__getitem__.assert_called_once_with(42)
