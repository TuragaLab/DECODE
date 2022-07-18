import numpy as np
from typing import Optional, Hashable, TypeVar

import torch

CodeType = TypeVar("CodeType")


class Code:
    def __init__(self, codes: list[int], rng: Optional[np.random.Generator] = None):
        """
        Sample codes

        Args:
            codes: list of codes to sample from
            rng: random number generator
        """
        self._codes = codes
        self._rng: np.random.Generator = (
            rng if rng is not None else np.random.default_rng()
        )

    def sample(self, n: int) -> torch.LongTensor:
        return torch.from_numpy(self._rng.choice(self._codes, size=n))


class CodeBook:
    def __init__(
        self,
        code_map: dict[Hashable, Hashable],
        rng: Optional[np.random.Generator] = None,
    ):
        """
        Codebook that holds a map from codes to bits

        Args:
            code_map: maps codes to bits
            rng: random number generator

        Examples:
            >>> c = CodeBook({0: [1, 0, 0, 1], 1: [0, 0, 0, 1]})
            >>> c.sample(5)
            [2, 2, 0, 4, 3]
        """
        code_map = (
            dict.fromkeys(code_map) if not isinstance(code_map, dict) else code_map
        )

        self.code_map = code_map
        self._code_map_inverse = {v: k for k, v in code_map.items()}

        self._rng: np.random.Generator = (
            rng if rng is not None else np.random.default_rng()
        )

        if len(self._code_map_inverse) != len(self.code_map):
            raise ValueError("Code map must be invertible.")

    def sample(self, n: int):
        return self.sample_codes(n)

    def sample_codes(self, n: int):
        return self._rng.choice(list(self.code_map.keys()), size=n)

    def sample_bits(self, n: int):
        return self._rng.choice(list(self.code_map.values()), size=n)

    def invert(self, bits: list[Hashable]) -> list:
        """Transform bits to codes."""
        return [self._code_map_inverse[b] for b in bits]
