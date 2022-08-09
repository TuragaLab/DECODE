import torch

from decode import emitter
from decode.emitter import process


def test_process_identity():
    em = emitter.factory(100)
    em_out = process.EmitterProcessNoOp().forward(em)

    assert em == em_out


def test_process_generic():
    f = process.EmitterFilterGeneric(phot=lambda p: p > 100, frame_ix=lambda f: f >= 5)
    em = emitter.factory(
        phot=torch.rand(100) * 100, frame_ix=torch.randint(10, size=(100,))
    )

    em_out = f.forward(em)
    assert em != em_out
    assert (em_out.phot > 100.0).all()
    assert (em_out.frame_ix >= 5).all()


def test_remove_out_of_field():
    em = emitter.factory(100000, extent=100)
    em.xyz[:, 2] = torch.rand_like(em.xyz[:, 2]) * 1500 - 750

    rmf = process.EmitterFilterFoV((0.0, 31.0), (7.5, 31.5), (-500, 700))
    em_out = rmf.forward(em)

    assert len(em_out) <= len(em)

    assert (em_out.xyz[:, 0] >= 0.0).all()
    assert (em_out.xyz[:, 1] >= 7.5).all()
    assert (em_out.xyz[:, 2] >= -500.0).all()

    assert (em_out.xyz[:, 0] < 31.0).all()
    assert (em_out.xyz[:, 1] < 31.5).all()
    assert (em_out.xyz[:, 2] < 700.0).all()
