import pytest
import h5py

from decode.generic import emitter
from decode.utils import emitter_io


def test_save_load_h5py(tmpdir):
    path = tmpdir / 'emitter.h5'

    em = emitter.RandomEmitterSet(20, xy_unit='px', px_size=(100, 200))
    emitter_io.save_h5(path, em.data, em.meta)

    data, meta = emitter_io.load_h5(path)
    em_h5 = emitter.EmitterSet(**data, **meta)
    assert em == em_h5  # if equality check is wrong, this is wrong as well
