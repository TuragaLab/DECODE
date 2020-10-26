import torch


def get_device_capability() -> str:
    capability = torch.cuda.get_device_capability()
    return f'{capability[0]}.{capability[1]}'
