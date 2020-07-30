import pytest

import torch
import numpy as np

from decode.simulation import structure_prior


class TestAbstractStructure:

    @pytest.fixture()
    def structure(self):

        class MockStructure(structure_prior.StructurePrior):

            @property
            def area(self):
                return 1.

            def sample(self, n: int) -> torch.Tensor:
                return torch.rand((n, 3))

        return MockStructure()

    def test_area(self, structure):

        assert structure.area >= 0.

    def test_sample(self, structure):

        assert structure.sample(100).size() == torch.Size((100, 3))


class TestRandomStructure(TestAbstractStructure):

    @pytest.fixture()
    def structure(self):
        return structure_prior.RandomStructure(xextent=(100., 200.),
                                               yextent=(1., 5.),
                                               zextent=(20., 30.))

    def test_area(self, structure):

        super().test_area(structure)

        assert structure.area == 100. * 4.

    def test_sample(self, structure):

        super().test_sample(structure)

        """Run"""
        xyz = structure.sample(100000)

        """Assert"""
        # test flatness of histogram
        hists = [np.histogram(xyz[:, i].numpy(), bins=100)[0] for i in range(3)]

        for h in hists:
            assert np.std(h) / np.mean(h) <= 0.1



