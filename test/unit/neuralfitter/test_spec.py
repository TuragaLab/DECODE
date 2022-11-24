import pytest

from decode.neuralfitter import spec


@pytest.mark.parametrize("n_code,n", [(1, 10), (2, 11), (4, 13)])
def test_spec_n(n_code, n):
    s = spec.ModelChannelMapGMM(n_codes=n_code)

    assert s.n == n
    assert s.n_prob == n_code
    assert s.n_mu == 4
    assert s.n_sig == 4
    assert s.n_bg == 1


@pytest.mark.parametrize("n_code,ix_prob,ix_mu,ix_sig,ix_bg,ix_phot,ix_xyz", [
    (1, [0], [1, 2, 3, 4], [5, 6, 7, 8], [9], [1], [2, 3, 4]),
    (3, [0, 1, 2], [3, 4, 5, 6], [7, 8, 9, 10], [11], [3], [4, 5, 6]),
])
def test_spec_ix(n_code, ix_prob, ix_mu, ix_sig, ix_bg, ix_phot, ix_xyz):
    s = spec.ModelChannelMapGMM(n_codes=n_code)

    assert s.ix_prob == ix_prob
    assert s.ix_mu == ix_mu
    assert s.ix_sig == ix_sig
    assert s.ix_bg == ix_bg
    assert s.ix_phot == ix_phot
    assert s.ix_xyz == ix_xyz
