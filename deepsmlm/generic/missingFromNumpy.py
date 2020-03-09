from functools import lru_cache, wraps
import math
import numpy as np
import os
import torch


def get_free_gpu(safety_factor=0.8):
    """
    https://discuss.pytorch.org/t/it-there-anyway-to-let-program-select-free-gpu-automatically/17560/7
    """
    if torch.cuda.is_available():
        os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
        memory_available = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
        return np.argmax(memory_available), np.max(memory_available) * 10**6 * safety_factor
    else:
        return None


def memory_of_tensor(a):
    return a.element_size() * a.nelement()


def splitbatchandrunfunc(b_vector, func, func_args, batch_size_target=None, to_cuda=False):
    if batch_size_target is None:
        b_vector_mem = memory_of_tensor(b_vector)
        _, free_mem = get_free_gpu()
        if free_mem > b_vector_mem:
            batch_size_target = b_vector.shape[0]
        else:
            batch_size_target = math.floor(free_mem / b_vector_mem * b_vector.shape[0])

    b_vector_split = torch.split(b_vector, batch_size_target)
    out = [None] * b_vector_split.__len__()
    for i in range(b_vector_split.__len__()):
        if to_cuda:
            out[i] = func(b_vector_split[i].cuda(), *func_args).cpu()
        else:
            out[i] = func(b_vector_split[i], *func_args)
    return torch.cat(out, 0)


def np_cache(*args, **kwargs):
    """LRU cache implementation for functions whose FIRST parameter is a numpy array
    >>> array = np.array([[1, 2, 3], [4, 5, 6]])
    >>> @np_cache(maxsize=256)
    ... def multiply(array, factor):
    ...     print("Calculating...")
    ...     return factor*array
    >>> multiply(array, 2)
    Calculating...
    array([[ 2,  4,  6],
           [ 8, 10, 12]])
    >>> multiply(array, 2)
    array([[ 2,  4,  6],
           [ 8, 10, 12]])
    >>> multiply.cache_info()
    CacheInfo(hits=1, misses=1, maxsize=256, currsize=1)

    """

    def decorator(function):
        @wraps(function)
        def wrapper(np_array, *args, **kwargs):
            hashable_array = array_to_tuple(np_array)
            return cached_wrapper(hashable_array, *args, **kwargs)

        @lru_cache(*args, **kwargs)
        def cached_wrapper(hashable_array, *args, **kwargs):
            array = np.array(hashable_array)
            return function(array, *args, **kwargs)

        def array_to_tuple(np_array):
            """Iterates recursivelly."""
            try:
                return tuple(array_to_tuple(_) for _ in np_array)
            except TypeError:
                return np_array

        # copy lru_cache attributes over too
        wrapper.cache_info = cached_wrapper.cache_info
        wrapper.cache_clear = cached_wrapper.cache_clear

        return wrapper

    return decorator
