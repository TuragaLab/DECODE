import torch

from decode import emitter
from decode.emitter import process


def test_remove_out_of_field():
    em = emitter.factory(100000, extent=100)
    em.xyz[:, 2] = torch.rand_like(em.xyz[:, 2]) * 1500 - 750

    rmf = process.RemoveOutOfFOV((0., 31.), (7.5, 31.5), (-500, 700))
    em_out = rmf.forward(em)

    assert len(em_out) <= len(em)

    assert (em_out.xyz[:, 0] >= 0.).all()
    assert (em_out.xyz[:, 1] >= 7.5).all()
    assert (em_out.xyz[:, 2] >= -500.).all()

    assert (em_out.xyz[:, 0] < 31.).all()
    assert (em_out.xyz[:, 1] < 31.5).all()
    assert (em_out.xyz[:, 2] < 700.).all()
