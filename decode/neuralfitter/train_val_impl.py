import torch
import time
from typing import Union

from tqdm import tqdm
from collections import namedtuple

from .utils import log_train_val_progress
from ..evaluation.utils import MetricMeter


def train(model, optimizer, loss, dataloader, grad_rescale, grad_mod, epoch, device, logger) -> float:

    """Some Setup things"""
    model.train()
    tqdm_enum = tqdm(dataloader, total=len(dataloader), smoothing=0.)  # progress bar enumeration
    t0 = time.time()
    loss_epoch = MetricMeter()

    """Actual Training"""
    for batch_num, (x, y_tar, weight) in enumerate(tqdm_enum):  # model input (x), target (yt), weights (w)

        """Monitor time to get the data"""
        t_data = time.time() - t0

        """Ship the data to the correct device"""
        x, y_tar, weight = ship_device([x, y_tar, weight], device)
        
        """Forward the data"""
        y_out = model(x)

        """Reset the optimiser, compute the loss and backprop it"""
        loss_val = loss(y_out, y_tar, weight)

        if grad_rescale:  # rescale gradients so that they are in the same order for the last layer
            weight, _, _ = model.rescale_last_layer_grad(loss_val, optimizer)
            loss_val = loss_val * weight

        optimizer.zero_grad()
        loss_val.mean().backward()

        """Gradient Modification"""
        if grad_mod:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.03, norm_type=2)

        """Update model parameters"""
        optimizer.step()

        """Monitor overall time"""
        t_batch = time.time() - t0

        """Logging"""
        loss_mean, loss_cmp = loss.log(loss_val)  # compute individual loss components
        del loss_val
        loss_epoch.update(loss_mean)
        tqdm_enum.set_description(f"E: {epoch} - t: {t_batch:.2} - t_dat: {t_data:.2} - L: {loss_mean:.3}")

        t0 = time.time()

    log_train_val_progress.log_train(loss_p_batch=loss_epoch.vals, loss_mean=loss_epoch.mean, logger=logger, step=epoch)

    return loss_epoch.mean


_val_return = namedtuple("network_output", ["loss", "x", "y_out", "y_tar", "weight", "em_tar"])


def test(model, loss, dataloader, epoch, device):

    """Setup"""
    x_ep, y_out_ep, y_tar_ep, weight_ep, em_tar_ep = [], [], [], [], []  # store things epoche wise (_ep)
    loss_cmp_ep = []

    model.eval()
    tqdm_enum = tqdm(dataloader, total=len(dataloader), smoothing=0.)  # progress bar enumeration

    t0 = time.time()

    """Testing"""
    with torch.no_grad():
        for batch_num, (x, y_tar, weight) in enumerate(tqdm_enum):

            """Ship the data to the correct device"""
            x, y_tar, weight = ship_device([x, y_tar, weight], device)

            """
            Forward the data.
            """
            y_out = model(x)

            loss_val = loss(y_out, y_tar, weight)

            t_batch = time.time() - t0

            """Logging and temporary save"""
            tqdm_enum.set_description(f"(Test) E: {epoch} - T: {t_batch:.2}")

            loss_cmp_ep.append(loss_val.detach().cpu())
            x_ep.append(x.cpu())
            y_out_ep.append(y_out.detach().cpu())

    """Epoch-Wise Merging"""
    loss_cmp_ep = torch.cat(loss_cmp_ep, 0)
    x_ep = torch.cat(x_ep, 0)
    y_out_ep = torch.cat(y_out_ep, 0)

    return loss_cmp_ep.mean(), _val_return(loss=loss_cmp_ep, x=x_ep, y_out=y_out_ep, y_tar=None, weight=None, em_tar=None)


def ship_device(x, device: Union[str, torch.device]):
    """
    Ships the input to a pytorch compatible device (e.g. CUDA)

    Args:
        x:
        device:

    Returns:
        x

    """
    if x is None:
        return x

    elif isinstance(x, torch.Tensor):
        return x.to(device)

    elif isinstance(x, (tuple, list)):
        x = [ship_device(x_el, device) for x_el in x]  # a nice little recursion that worked at the first try
        return x

    elif device != 'cpu':
        raise NotImplementedError(f"Unsupported data type for shipping from host to CUDA device.")
