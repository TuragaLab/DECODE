import copy

import pytest
import torch

from decode.generic import test_utils
from decode.neuralfitter import loss
from decode.neuralfitter import post_processing
from decode.neuralfitter import train_val_impl
from decode.neuralfitter.utils import logger as logger_utils


class TestTrain:

    @pytest.fixture()
    def model_mock(self):
        class MockModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.a = torch.nn.Parameter(torch.Tensor([5]), requires_grad=True)

            def forward(self, x):
                return self.a * torch.rand(x.size(0), 6, x.size(2), x.size(3), device=x.device)

        return MockModel()

    @pytest.fixture()
    def opt(self, model_mock):
        return torch.optim.Adam(model_mock.parameters())

    @pytest.fixture()
    def loss(self):
        return loss.PPXYZBLoss(torch.device('cpu'))

    @pytest.fixture()
    def dataloader(self):
        class MockDataset(torch.utils.data.Dataset):
            def __len__(self):
                return 256

            def __getitem__(self, item):
                return torch.rand((1, 64, 64)), torch.rand((6, 64, 64)), torch.rand((6, 64, 64))

        return torch.utils.data.DataLoader(MockDataset(), batch_size=32)

    @pytest.fixture()
    def logger(self):
        return logger_utils.NoLog()

    @pytest.fixture()
    def train_val_environment(self, loss, model_mock):
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        # ship the things to the apropriate device
        loss.__init__(device=device)
        model = model_mock.to(device)

        return device, model

    @pytest.mark.skip("Segfault for low memory machines.")
    def test_iterate_batch(self, opt, loss, dataloader, logger, train_val_environment):
        device, model = train_val_environment

        model_before = copy.deepcopy(model)

        """Run"""
        # ToDo: The following line causes segmentation fault on low memory machines
        train_val_impl.train(model, opt, loss, dataloader, False, False, 0, device, logger)

        assert not test_utils.same_weights(model_before, model)


class TestVal(TestTrain):

    @pytest.fixture()
    def post_processor(self):
        return post_processing.ConsistencyPostprocessing(raw_th=0.1, em_th=0.5, xy_unit='nm', img_shape=(64, 64),
                                                         lat_th=100)

    def test_iterate_batch(self, opt, loss, dataloader, post_processor, logger, train_val_environment):
        device, model = train_val_environment

        model_before = copy.deepcopy(model)

        """Run"""
        train_val_impl.test(model, loss, dataloader, 0, device)

        assert test_utils.same_weights(model_before, model)
