from abc import ABC, abstractmethod
from typing import Protocol, runtime_checkable


class ModelChannelMap(ABC):
    """
    Helper to map model output to semantic channels.
    """
    @property
    @abstractmethod
    def n(self) -> int:
        raise NotImplementedError


@runtime_checkable
class _ModelChannelMapInferredAttr(Protocol):
    # just for linting
    ix_phot_sig: list[int]
    ix_xyz_sig: list[int]


class ModelChannelMapGMM(_ModelChannelMapInferredAttr, ModelChannelMap):
    _delta_mu_sig = 4  # delta ix between mu and sigma

    def __init__(self, n_codes: int):
        """
        Helper to map model output of gaussian mixture model to semantic channels.

        Args:
            n_codes:
        """
        self._n_codes = n_codes
        self._ix = list(range(self.n))

    @property
    def n(self) -> int:
        return self.n_prob + self.n_mu + self.n_sig + self.n_bg

    @property
    def n_prob(self) -> int:
        return self._n_codes

    @property
    def n_mu(self) -> int:
        return 4

    @property
    def n_sig(self) -> int:
        return 4

    @property
    def n_bg(self) -> int:
        return 1

    @property
    def ix_prob(self) -> list[int]:
        return self._ix[:self.n_prob]

    @property
    def ix_mu(self) -> list[int]:
        return self._ix[self.n_prob:(self.n_prob + self.n_mu)]

    @property
    def ix_sig(self) -> list[int]:
        return self._ix[(self.n_prob + self.n_mu):(self.n_prob + self.n_mu + self.n_sig)]

    @property
    def ix_bg(self) -> list[int]:
        return self._ix[-1:]

    @property
    def ix_phot(self) -> list[int]:
        return self.ix_mu[:1]

    @property
    def ix_xyz(self) -> list[int]:
        return self.ix_mu[1:]

    def __getattr__(self, item):

        # _sig attributes are inferred by delta between mu and sigma
        if "_sig" in item and hasattr(self, item.rstrip("_sig")):
            return [mu + self._delta_mu_sig for mu in getattr(self, item.rstrip("_sig"))]

        raise AttributeError
