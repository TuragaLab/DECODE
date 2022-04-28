import torch
import pytest

import decode.generic.slicing as gutils  # candidate


class TestSplitSliceable:

    @staticmethod
    def split_sliceable(x, x_ix, ix_low, ix_high):
        """
        Simple, equivalent but slow function for testing comparison.

        Args:
            x: sliceable / iterable
            x_ix (torch.Tensor): index according to which to split
            ix_low (int): lower bound
            ix_high (int): upper bound

        Returns:
            x_list: list of instances sliced as specified by the x_ix

        """

        out = []
        for i in range(ix_low, ix_high):
            out.append(x[x_ix == i])

        return out

    @pytest.mark.parametrize("x,x_ix,ix_low,ix_high", [
        ([1, 2, 3], [5, 3, 1], 17, 21)
    ])
    def test_equivalence(self, x, x_ix, ix_low, ix_high):
        x = torch.Tensor(x)
        x_ix = torch.LongTensor(x_ix)

        out = gutils.split_sliceable(x, x_ix, ix_low, ix_high)
        exp = self.split_sliceable(x, x_ix, ix_low, ix_high)

        assert len(out) == len(exp) == ix_high - ix_low, "Incorrect length."
        for o, e in zip(out, exp):
            assert (o == e).all(), "Fast and slow implementation are not equivalent."

    def test_exceptions(self):

        # non integer index
        with pytest.raises(TypeError):
            gutils.split_sliceable(torch.tensor([0.]), torch.tensor([1.2]), 0, 1)

        # unequal length of index and slicable
        with pytest.raises((IndexError, ValueError)):
            gutils.split_sliceable(torch.tensor([]), torch.tensor([1]), 0, 1)

        # unequal length of index and slicable
        with pytest.raises((IndexError, ValueError)):
            gutils.split_sliceable(torch.tensor([1.]), torch.tensor([]), 0, 1)

    empty_cases = [
        (torch.tensor([]), torch.tensor([]), 0, 0, []),
        (torch.tensor([]), torch.tensor([]), 0, 1, [torch.tensor([])]),
        (torch.tensor([]), torch.tensor([]), 0, -1, [])
    ]

    @pytest.mark.parametrize("x,ix,ix_low,ix_high,exp", empty_cases)
    def test_empty_cases(self, x, ix, ix_low, ix_high, exp):
        out = gutils.split_sliceable(x, ix, ix_low, ix_high)
        exp_native = self.split_sliceable(x, ix, ix_low, ix_high)

        assert len(out) == len(exp_native) == len(exp), "Lenghts not equal."

        for o, e, e_nat in zip(out, exp, exp_native):
            assert (o == e).all(), "Implementation and manual expectation not equal."
            assert (o == e_nat).all(), "Implementation and native implementation not equal."

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


# ToDo: This needs a more thorough test
@pytest.mark.parametrize("ix,ix_low,ix_high,exp", [
    ([2, -1, 2, 0, 4], -1, 4, [[-1], [0], [], [2, 2], []])
])
def test_ix_splitting(ix, ix_low, ix_high, exp):
    ix = torch.LongTensor(ix)
    exp = [torch.LongTensor(e) for e in exp]

    out, n = gutils.ix_split(ix, ix_low, ix_high)

    assert n == len(out) == len(exp), "Length incorrect"

    for o, e in zip(out, exp):
        # hacky test. using the boolean output to boolean access the input index
        assert (ix[o] == e).all(), "Elements unequal."
