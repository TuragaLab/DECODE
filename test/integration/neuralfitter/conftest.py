import pytest

from . import common


@pytest.fixture
def mock_model_gmm() -> common.MockModelGMM:
    return common.MockModelGMM()


@pytest.fixture
def mock_loss_gmm() -> common.MockLossGMM:
    return common.MockLossGMM()
