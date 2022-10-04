from typing import Any, Protocol


class Experiment(Protocol):
    def sample(self) -> Any:
        ...
