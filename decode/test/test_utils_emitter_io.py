import torch
import pytest
from unittest import mock

from decode.generic import emitter
from decode.utils import emitter_io


@pytest.fixture()
def em_rand():
    return emitter.RandomEmitterSet(20, xy_unit='px', px_size=(100, 200))


@pytest.fixture()
def em_all_attrs(em_rand):
    em_rand = em_rand.clone()
    em_rand.xyz_sig = torch.rand(20, 3)
    em_rand.xyz_cr = torch.rand_like(em_rand.xyz_sig)
    em_rand.phot_sig = torch.rand(20)
    em_rand.phot_cr = torch.rand_like(em_rand.phot_sig)
    em_rand.bg_sig = torch.rand(20)
    em_rand.bg_cr = torch.rand_like(em_rand.bg_sig)

    return em_rand


@pytest.mark.parametrize("save_fn,load_fn,extension", [
    (emitter_io.save_h5, emitter_io.load_h5, '.h5'),
    (emitter_io.save_torch, emitter_io.load_torch, '.pt'),
    (emitter_io.save_csv, emitter_io.load_csv, '.csv')
])
def test_save_load_h5py(em_rand, em_all_attrs, save_fn, load_fn, extension, tmpdir):
    path = str(tmpdir / f'emitter{extension}')

    for em in (em_rand, em_all_attrs):
        save_fn(path, em.data, em.meta)

        data, meta, decode_meta = load_fn(path)
        em_reloaded = emitter.EmitterSet(**data, **meta)

        assert em == em_reloaded  # if equality check is wrong, this is wrong as well
        assert decode_meta['version'][0] == 'v'


@pytest.mark.parametrize('last_index', ['including', 'excluding'])
def test_streamer(last_index, tmpdir):

    stream = emitter_io.EmitterWriteStream('dummy', '.pt', tmpdir, last_index=last_index)

    with mock.patch.object(emitter.EmitterSet, 'save') as mock_save:
        stream.write(emitter.RandomEmitterSet(20), 0, 100)

    if last_index == 'including':
        mock_save.assert_called_once_with(tmpdir / 'dummy_0_100.pt')
    elif last_index == 'excluding':
        mock_save.assert_called_once_with(tmpdir / 'dummy_0_99.pt')

    with mock.patch.object(emitter.EmitterSet, 'save') as mock_save:
        stream(emitter.RandomEmitterSet(20), 0, 100)

    mock_save.assert_called_once()
