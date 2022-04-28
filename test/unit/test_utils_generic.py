import torch
import pytest

import decode.generic.slicing as gutils  # candidate


class TestSplitSliceable:

    @staticmethod
    def split_sliceable(x, x_ix, ix_low, ix_high):
        """
        Mock function for testing. Slow but works definitely.

        Args:
            x: sliceable / iterable
            x_ix (torch.Tensor): index according to which to split
            ix_low (int): lower bound
            ix_high (int): upper bound

        Returns:
            x_list: list of instances sliced as specified by the x_ix

        """

        out = []
        for i in range(ix_low, ix_high + 1):
            out.append(x[x_ix == i])

        return out

    def test_exceptions(self):

        """Assert"""
        with pytest.raises(TypeError):
            gutils.split_sliceable(torch.tensor([0.]), torch.tensor([1.2]), 0, 1)  # non integer index

        with pytest.raises((IndexError, ValueError)):
            gutils.split_sliceable(torch.tensor([]), torch.tensor([1]), 0, 1)  # non integer index

        with pytest.raises((IndexError, ValueError)):
            gutils.split_sliceable(torch.tensor([1.]), torch.tensor([]), 0, 1)  # non integer index

    border_cases = [
        (torch.tensor([]), torch.tensor([]), 0, 0, [torch.tensor([])]),
        (torch.tensor([]), torch.tensor([]), 0, 1, [torch.tensor([]), torch.tensor([])]),
        (torch.tensor([]), torch.tensor([]), 0, -1, [])
    ]

    @pytest.mark.parametrize("x,ix,ix_low,ix_high,exp", border_cases)
    def test_empty(self, x, ix, ix_low, ix_high, exp):
        """
        Wiggling with border cases
        """

        """Run"""
        out = gutils.split_sliceable(x, ix, ix_low, ix_high)

        """Asserts"""
        for o, e in zip(out, exp):
            assert (o == e).all()

    small_cases = [
        # supposed to called from max. 0 to 3
        (torch.tensor([[1., 2.], [3., 4.], [-1., 2.], [7., 8.], [-5., 5.]]), torch.tensor([1, 0, 2, 2, 5]), [
            torch.tensor([[3., 4.]]), torch.tensor([[1., 2.]]), torch.tensor([[-1., 2.], [7., 8.]]), torch.empty((0, 2))
        ])
    ]

    @pytest.mark.parametrize("x,ix,exp", small_cases)
    @pytest.mark.parametrize("ix_low", [0, 1, 2, 3])
    @pytest.mark.parametrize("ix_high", [0, 1, 2, 3])
    def test_handcrafted(self, x, ix, ix_low, ix_high, exp):
        """Handcrafted cases: """

        """Run"""
        out = gutils.split_sliceable(x, ix, ix_low, ix_high)
        out_ref = self.split_sliceable(x, ix, ix_low, ix_high)

        """Asserts"""
        for o, e in zip(out, exp[ix_low:(ix_high + 1)]):  # pick apropriate elements from the expected results
            assert (o == e).all()

        # compare to simple reference implementation
        for o, oref in zip(out, out_ref):
            assert (o == oref).all()


def test_ix_splitting():
    ix = torch.Tensor([2, -1, 2, 0, 4]).int()

    out, n = gutils.ix_split(ix, -1, 4)

    assert len(out) == 6
    assert len(out) == n
    assert ix[out[0]] == -1
    assert ix[out[1]] == 0
    assert ix[out[2]].numel() == 0
    assert (ix[out[3]] == 2).all()
    assert ix[out[4]].numel() == 0
    assert ix[out[5]] == 4