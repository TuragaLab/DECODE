import torch


class MockModelGMM(torch.nn.Module):
    def __init__(self, ch_in: int = 3, ch_out: int = 10):
        super().__init__()
        self._ch_in = ch_in
        self._ch_out = ch_out

    def verify_input(self, x: torch.Tensor):
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.verify_input(x)

        return torch.rand(x.size(0), self._ch_out, x.size(-2), x.size(-1))


class MockLossGMM(torch.nn.Module):
    def __init__(self, n_codes: int = 1):
        super().__init__()
        self._n_codes = n_codes

    def verify_input(self, x: torch.Tensor):
        if x.size() != self._n_codes + 9:
            raise ValueError("Incorrect number of channels")

    def forward(self, output: torch.Tensor, target: tuple, weight) -> torch.Tensor:
        self.verify_input(output)

        return torch.tensor(42)
