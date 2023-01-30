import torch


def _get_device_types_available():
    d = [torch.device("cpu")]
    d.append(torch.device("cuda:0")) if torch.cuda.is_available() else ...
    d.append(torch.device("mps")) if torch.backends.mps.is_available() else ...

    return d

