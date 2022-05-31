import numpy as np
from typing import Optional, Hashable


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
        """
        self.code_map = code_map
        self._code_map_inverse = {v: k for k, v in code_map.items()}

        self._rng: np.random.Generator = (
            rng if rng is not None else np.random.default_rng()
        )

        if len(self._code_map_inverse) != len(self.code_map):
            raise ValueError("Code map must be invertible.")

    def sample_codes(self, n: int):
        return self._rng.choice(list(self.code_map.keys()), size=n)

    def sample_bits(self, n: int):
        return self._rng.choice(list(self.code_map.values()), size=n)

    def invert(self, bits: list[Hashable]) -> list:
        """Transform bits to codes."""
        return [self._code_map_inverse[b] for b in bits]
