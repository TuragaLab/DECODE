import torch
import torch.utils.data

import decode.generic


def smlm_collate(batch):
    """
    Collate for dataloader that allows for None return and EmitterSet.
    Otherwise defaults to default pytorch collate

    Args:
        batch
    """
    elem = batch[0]
    # BEGIN PARTLY INSERTION of default collate
    if isinstance(elem, torch.Tensor):
        out = None
        if torch.utils.data.get_worker_info() is not None:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum([len(x.view(-1)) for x in batch])
            storage = elem.storage()._new_shared(numel)
            out = elem.new(storage)
            out = out.reshape(len(batch), *batch[0].shape)
        return torch.stack(batch, 0, out=out)
    elif isinstance(elem, (list, tuple)):
        # check to make sure that the elements in batch have consistent size
        it = iter(batch)
        elem_size = len(next(it))
        if not all(len(elem) == elem_size for elem in it):
            raise RuntimeError("each element in list of batch should be of equal size")
        transposed = zip(*batch)
        return [smlm_collate(samples) for samples in transposed]
    # END INSERT
    elif elem is None:
        return None
    elif isinstance(elem, decode.generic.emitter.EmitterSet):
        return [em for em in batch]
    else:
        return torch.utils.data.dataloader.default_collate(batch)
