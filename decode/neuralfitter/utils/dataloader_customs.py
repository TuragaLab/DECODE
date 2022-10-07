import torch
import torch.utils.data
from deprecated import deprecated

from ... import emitter


@deprecated(reason="Code duplication and not necessary anymore.", version="0.11.0")
def smlm_collate(batch):
    """
    Collate for dataloader that allows for None return and EmitterSet.
    Otherwise defaults to default pytorch collate

    Args:
        batch
    """
    elem = batch[0]
    # ToDo: possible rewrite? One must break out of recursion
    # BEGIN PARTLY INSERTION of default collate
    if isinstance(elem, torch.Tensor):
        out = None
        if torch.utils.data.get_worker_info() is not None:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum([x.numel() for x in batch])
            storage = elem.storage()._new_shared(numel)
            out = elem.new(storage)
        return torch.stack(batch, 0, out=out)
    elif isinstance(elem, (list, tuple)):
        # check to make sure that the elements in batch have consistent size
        it = iter(batch)
        elem_size = len(next(it))
        if not all(len(elem) == elem_size for elem in it):
            raise RuntimeError('each element in list of batch should be of equal size')
        transposed = zip(*batch)
        return [smlm_collate(samples) for samples in transposed]
    # END INSERT
    elif elem is None:
        return None
    elif isinstance(elem, emitter.EmitterSet):
        return [em for em in batch]
    else:
        return torch.utils.data.dataloader.default_collate(batch)
