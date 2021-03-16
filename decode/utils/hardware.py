from typing import Tuple, Union

import torch


def _specific_device_by_str(device) -> Tuple[str, str]:
    """Converts torch compatible device string to device name and device index"""
    if device != 'cpu' and device[:4] != 'cuda':
        raise ValueError

    if device == 'cpu':
        return 'cpu', None

    elif len(device) == 4:
        return 'cuda', None

    else:
        return 'cuda', int(device.split(':')[-1])


def get_device_capability() -> str:
    capability = torch.cuda.get_device_capability()
    return f'{capability[0]}.{capability[1]}'


def get_max_batch_size(callable, x_size: Tuple, device: Union[str, torch.device], size_low: int, size_high: int):
    if size_low > size_high:
        raise ValueError("Lower bound must be lower than upper bound.")

    bs = size_low
    bs_fail = size_high
    bs_pass = None

    while bs < size_high:

        try:
            x_try = torch.rand(bs, *x_size, device=device)
            callable(x_try)
            bs_pass = bs

        except RuntimeError as err:
            if 'CUDA out of memory.' not in str(err):
                raise err

            bs_fail = bs

        if bs_pass is None:
            raise RuntimeError("Lowest possible batch size is outside of provided bounds.")

        bs_new = int((bs_fail + bs_pass) / 2)

        if bs_new == bs:
            break

        bs = bs_new

    del x_try
    torch.cuda.empty_cache()
    return bs
