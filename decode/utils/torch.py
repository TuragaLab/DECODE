import torch


class ItemizedDist:
    def __init__(self, dist: torch.distributions.Distribution):
        """Helper class to output single values from torch dists"""
        self._dist = dist

    def __call__(self):
        return self.sample()

    def sample(self):
        return self._dist.sample((1, )).item()
