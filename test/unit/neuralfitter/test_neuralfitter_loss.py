import pytest
import torch

from decode.generic import test_utils
from decode.neuralfitter import loss


class TestLossAbstract:

    @pytest.fixture()
    def loss_impl(self):
        class MockLoss(loss.Loss):
            """Mock loss. Assumes 2 channels."""

            def __init__(self):
                super().__init__()
                self._loss_impl = torch.nn.MSELoss(reduction='none')

            def log(self, loss_val):
                loss_vec = loss_val.mean(-1).mean(-1).mean(0)
                return loss_vec.mean().item(), {
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

        return random_loss_input[0].cuda(), random_loss_input[1].cuda(), random_loss_input[2].cuda()

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

    def test_log(self, loss_impl, random_loss_input):
        """
        Tests the return of the log implementation of the loss implementation
        """

        mean, components = loss_impl.log(random_loss_input[0])
        assert isinstance(mean, float)
        assert isinstance(components, dict)

        for log_el in components.values():
            assert isinstance(log_el, float)


class TestPPXYZBLoss(TestLossAbstract):

    @pytest.fixture()
    def loss_impl(self):
        return loss.PPXYZBLoss(device=torch.device('cpu'))

    @pytest.fixture(params=[1, 2, 64])
    def random_loss_input(self, request):
        return (torch.rand((request.param, 6, 64, 64)),
                torch.rand((request.param, 6, 64, 64)),
                torch.rand((request.param, 6, 64, 64)))

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available on test machine.")
    def test_forward_cuda(self, loss_impl, random_cuda):
        loss_impl.__init__(device=torch.device('cuda'))  # re-init with cuda
        super().test_forward_cuda(loss_impl, random_cuda)

    def test_forward_quant(self, loss_impl):
        """Run and Assert"""
        # all zero
        assert (torch.zeros((2, 6, 32, 32)) == loss_impl.forward(*([torch.zeros((2, 6, 32, 32))] * 3))).all()

        # check ch weight
        loss_ch = loss.PPXYZBLoss(device=torch.device('cpu'), chweight_stat=(1., 2., 1., 1., 1., 1.))
        out = loss_ch.forward(torch.zeros((2, 6, 32, 32)), torch.ones((2, 6, 32, 32)), torch.ones((2, 6, 32, 32)))

        assert test_utils.tens_almeq(out[:, 2:], torch.ones_like(out[:, 2:]))
        assert test_utils.tens_almeq(out[:, 1], torch.ones_like(out[:, 1]) * 2)


class TestGaussianMixtureModelLoss:

    @pytest.fixture()
    def loss_impl(self):
        return loss.GaussianMMLoss(xextent=(-0.5, 31.5), yextent=(-0.5, 31.5), img_shape=(32, 32), device='cpu')

    @pytest.fixture()
    def data_handcrafted(self):

        p = torch.zeros((2, 32, 32)) + 1E-8
        pxyz_mu = torch.zeros((2, 4, 32, 32))
        pxyz_sig = torch.zeros_like(pxyz_mu) + 100

        p[[0, 1], [2, 5], [4, 10]] = 0.9
        pxyz_mu[0, :, 2, 4] = torch.tensor([0.76, 0.3, -0.7, 0.8])
        pxyz_mu[1, :, 5, 10] = torch.tensor([0.8, 0.2, 0.4, 0.1])
        pxyz_sig[0, :, 2, 4] = torch.tensor([3., 1., 0.5, 0.2])
        pxyz_sig[1, :, 5, 10] = torch.tensor([.1, 2., 3., 4.])

        pxyz_tar = torch.zeros((2, 3, 4))
        pxyz_tar[0, 0, :] = torch.tensor([0.8, 2.2, 3.9, 0.7])
        pxyz_tar[1, 0, :] = torch.tensor([0.8, 4.1, 10.2, 0.05])

        mask = torch.zeros((2, 3)).long()
        mask[[0, 1], 0] = 1

        return mask, p, pxyz_mu, pxyz_sig, pxyz_tar

    def test_gmm_loss(self, loss_impl, data_handcrafted):
        mask, p, pxyz_mu, pxyz_sig, pxyz_tar = data_handcrafted

        """Run"""
        out = loss_impl._compute_gmm_loss(p, pxyz_mu.requires_grad_(True), pxyz_sig, pxyz_tar, mask)
        out.sum().backward()

    def test_loss_forward_backward(self, loss_impl, data_handcrafted):

        """Setup"""
        mask, p, pxyz_mu, pxyz_sig, pxyz_tar = data_handcrafted
        bg_tar = torch.rand((2, 32, 32))

        model_out = torch.cat((p.unsqueeze(1), pxyz_mu, pxyz_sig, torch.rand((2, 1, 32, 32))), 1)
        model_out = model_out.clone().requires_grad_(True)

        """Run"""
        loss_val = loss_impl.forward(model_out, (pxyz_tar, mask, bg_tar), None)

        loss_val.mean().backward()

    def test_ch_static_weight(self, loss_impl, data_handcrafted):

        """Setup"""
        mask, p, _, _, pxyz_tar = data_handcrafted
        x = torch.rand((2, 9, 32, 32))
        bg_tar = torch.zeros((2, 32, 32))
        bg_out = torch.rand_like(bg_tar).unsqueeze(1)

        model_out = torch.cat((x, bg_out), 1).requires_grad_(True)

        """Run and Assert"""
        # all ch on
        loss_impl._ch_weight = torch.tensor([1., 1.])
        loss_val = loss_impl.forward(model_out, (pxyz_tar, mask, bg_tar), None)
        _, log_out = loss_impl.log(loss_val)

        assert (loss_val != 0.).all()
        assert log_out['gmm'] != 0.
        assert log_out['bg'] != 0.

        # bg ch off
        loss_impl._ch_weight = torch.tensor([1., 0.])
        loss_val = loss_impl.forward(model_out, (pxyz_tar, mask, bg_tar), None)
        _, log_out = loss_impl.log(loss_val)

        assert (loss_val[:, 0] != 0.).all()
        assert (loss_val[:, 1] == 0.).all()
        assert log_out['gmm'] != 0.
        assert log_out['bg'] == 0.

        # gmm ch off
        loss_impl._ch_weight = torch.tensor([0., 1.])
        loss_val = loss_impl.forward(model_out, (pxyz_tar, mask, bg_tar), None)
        _, log_out = loss_impl.log(loss_val)

        assert (loss_val[:, 1] != 0.).all()
        assert (loss_val[:, 0] == 0.).all()
        assert log_out['gmm'] == 0.
        assert log_out['bg'] != 0.

