import pytest


class TestNoiseDistribution:
    @pytest.fixture
    def noise(self):
        raise NotImplementedError

    def test_forward(self, noise):
        raise NotImplementedError


class TestZeroNoise(TestNoiseDistribution):
    pass


class TestGaussian(TestNoiseDistribution):
    pass


class TestGamma(TestNoiseDistribution):
    pass


class TestPoisson(TestNoiseDistribution):
    pass
