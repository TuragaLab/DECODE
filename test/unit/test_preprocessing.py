import torch

import decode.generic.test_utils as tutil
import decode.generic.process as prep

from decode.generic.emitter import RandomEmitterSet


def test_identity():
    x = torch.rand((32, 32))

    assert tutil.tens_almeq(prep.Identity().forward(x), x)


def test_remove_out_of_field():
    # Setup
    em = RandomEmitterSet(100000, extent=100)
    em.xyz[:, 2] = torch.rand_like(em.xyz[:, 2]) * 1500 - 750

    # Candidate
    rmf = prep.RemoveOutOfFOV((0., 31.), (7.5, 31.5), (-500, 700))

    # Run and Test
    em_out = rmf.forward(em)

    assert len(em_out) <= len(em)

    assert (em_out.xyz[:, 0] >= 0.).all()
    assert (em_out.xyz[:, 1] >= 7.5).all()
    assert (em_out.xyz[:, 2] >= -500.).all()

    assert (em_out.xyz[:, 0] < 31.).all()
    assert (em_out.xyz[:, 1] < 31.5).all()
    assert (em_out.xyz[:, 2] < 700.).all()
