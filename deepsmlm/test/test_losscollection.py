import pytest
import torch

from deepsmlm.generic.utils import test_utils
from deepsmlm.neuralfitter import losscollection as loss


class TestLossAbstract:

    @pytest.fixture()
    def loss_impl(self):
        class MockLoss(loss.Loss):
            """Mock loss. Assumes 2 channels."""

            def __init__(self):
                super().__init__()
                self._loss_impl = torch.nn.MSELoss(reduction='none')

            def log(self, loss_val) -> dict:
                loss_vec = loss_val.mean(-1).mean(-1).mean(0)
                return {
                    'p': loss_vec[0].item(),
                    'x': loss_vec[1].item()
                }

            def forward(self, output, target, weight):
                self._forward_checks(output, target, weight)
                return self._loss_impl(output, target) * weight

        return MockLoss()

    @pytest.fixture(params=[1, 2, 64])
    def random_loss_input(self, request):
        """
        Random input that should work dimensional-wise but does not have to make sense in terms of values

        Returns:
            tuple: output, target, weight
        """
        return (torch.rand((request.param, 2, 64, 64)),
                torch.rand((request.param, 2, 64, 64)),
                torch.rand((request.param, 2, 64, 64)))

    @pytest.fixture()
    def random_cuda(self, random_loss_input):
        """
        Random cuda input

        Args:
            random_input: fixture as above
        """

        return (random_loss_input[0].cuda(), random_loss_input[1].cuda(), random_loss_input[2].cuda())

    def test_call(self, loss_impl, random_loss_input):

        assert (loss_impl(*random_loss_input) == loss_impl.forward(*random_loss_input)).all(), "Call does not yield " \
                                                                                               "same results"

    def test_forward_qual(self, loss_impl, random_loss_input):
        """
        Qualitative tests for forward implementation, i.e. shape etc.

        """

        with pytest.raises(ValueError):  # output shape mismatch
            loss_impl.forward(torch.rand((1, 2, 64, 64)), torch.rand((2, 2, 64, 64)), torch.rand((2, 2, 64, 64)))

        with pytest.raises(ValueError):  # target shape mismatch
            loss_impl.forward(torch.rand((2, 2, 64, 64)), torch.rand((1, 2, 64, 64)), torch.rand((2, 2, 64, 64)))

        with pytest.raises(ValueError):  # weight shape mismatch
            loss_impl.forward(torch.rand((2, 2, 64, 64)), torch.rand((2, 2, 64, 64)), torch.rand((1, 2, 64, 64)))

        _ = loss_impl(*random_loss_input)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available on test machine.")
    def test_forward_cuda(self, loss_impl, random_cuda):
        """Tests the CUDA capability"""

        _ = loss_impl(*random_cuda)

    def test_log(self, loss_impl):
        """
        Tests the return of the log implementation of the loss implementation
        """

        out = loss_impl.log(torch.rand((2, 2, 64, 64)))
        assert isinstance(out, dict)

        # ToDo: Check that the output is a scalar
        for log_el in out.values():
            assert isinstance(log_el, float)


class TestPPXYZBLoss(TestLossAbstract):

    @pytest.fixture()
    def loss_impl(self):
        return loss.PPXYZBLoss()

    @pytest.fixture(params=[1, 2, 64])
    def random_loss_input(self, request):
        return (torch.rand((request.param, 6, 64, 64)),
                torch.rand((request.param, 6, 64, 64)),
                torch.rand((request.param, 6, 64, 64)))


"""Deprecated stuff"""
# from tensorboardX import SummaryWriter


@pytest.mark.skip(reason="Deprecated implementation.")
class TestFocalLoss:

    @pytest.fixture(scope='class')
    def focal(self):
        return loss.FocalLoss(focusing_param=2, balance_param=1.)

    @pytest.fixture(scope='class')
    def ce_loss(self):
        return torch.nn.CrossEntropyLoss()

    def test(self, ce_loss):
        gt = torch.zeros((2, 32)).type(torch.LongTensor)
        gt[0, 0] = 1

        prediction = torch.zeros((2, 2, 32))
        prediction[0, 0, 0] = 1.0
        prediction[0, 1, 0] = 1.0

        focal = loss.FocalLoss(focusing_param=0., balance_param=1.)
        l = focal.forward(prediction, gt)
        l_ce = ce_loss.forward(prediction, gt)

        assert test_utils.tens_almeq(l, l_ce)

        focal_gamma2 = loss.FocalLoss(focusing_param=2.)
        l_2 = focal_gamma2.forward(prediction, gt)

        assert True


@pytest.mark.skip(reason="Deprecated implementation.")
class TestSpeiserLogged:

    @pytest.fixture(scope='class')
    def crit(self):
        # logger = SummaryWriter('temp', comment='dummy', write_to_disk=False)
        logger = None
        return loss.SpeiserLoss(False, logger=logger)

    def test_forward(self, crit):
        for _ in range(5):
            x = torch.rand((2, 5, 32, 32), requires_grad=True)
            gt = torch.rand_like(x)

            loss_ = crit(x, gt)
            loss_.mean().backward()

            crit.log_batch_loss_cmp(loss_)

        crit.log_components(1)
        assert True


@pytest.mark.skip(reason="Deprecated implementation.")
class TestOffsetROILoss:

    @pytest.fixture(scope='class')
    def loss(self):
        return loss.OffsetROILoss()

    def test_run(self, loss):
        x = torch.zeros((2, 5, 32, 32))
        x[0, 0, 1, 1] = 1.
        assert test_utils.tens_almeq(x, loss(x, x))


@pytest.mark.skip(reason="Deprecated implementation.")
class TestFocalVoronoiPointLoss:

    @pytest.fixture(scope='class')
    def fvp_loss(self):
        return loss.FocalVoronoiPointLoss(0.012, 0.9)

    @pytest.mark.skip("Skip.")
    def test_run(self, fvp_loss):
        gt = torch.zeros((2, 1, 32, 32))
        gt[0, 0, 0, 0] = 1

        prediction = gt.clone()
        prediction[0, 0, 0, 0] = 0.01

        loss = fvp_loss(prediction, gt)

        assert True
