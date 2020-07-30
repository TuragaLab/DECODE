import torch
from torch._six import int_classes, string_classes, container_abcs

import decode.generic


def smlm_collate(batch):
    error_msg = "batch must contain tensors, numbers, dicts or lists; found {}"
    # elem_type = type(batch[0])
    if isinstance(batch[0], torch.Tensor):
        out = None
        if True is True:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum([x.numel() for x in batch])
            storage = batch[0].storage()._new_shared(numel)
            out = batch[0].new(storage)
        return torch.stack(batch, 0, out=out)
    elif isinstance(batch[0], int_classes):
        return torch.LongTensor(batch)
    elif isinstance(batch[0], float):
        return torch.DoubleTensor(batch)
    elif isinstance(batch[0], string_classes):
        return batch
    elif isinstance(batch[0], container_abcs.Mapping):
        return {key: smlm_collate([d[key] for d in batch]) for key in batch[0]}
    elif isinstance(batch[0], container_abcs.Sequence):
        transposed = zip(*batch)
        return [smlm_collate(samples) for samples in transposed]
    elif isinstance(batch[0], decode.generic.emitter.EmitterSet):
        return [em for em in batch]
    elif batch[0] is None:
        # warnings.warn("Encountered 'None' variable returned from dataset.")
        return None
    else:
        raise TypeError((error_msg.format(type(batch[0]))))