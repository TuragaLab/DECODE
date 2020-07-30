import torch

import decode
from ..generic import test_utils
from ..utils import emitter_trafo


def test_emitter_transformation():
    """Setup"""
    em = decode.RandomEmitterSet(200, extent=10000, xy_unit='px', px_size=(100., 100.))
    em.frame_ix = torch.randint_like(em.frame_ix, 1, 10000)

    """Run"""
    mod_em = emitter_trafo.transform_emitter(em, emitter_trafo.challenge_import)

    """Assert"""
    assert id(em) != id(mod_em), "Emitter transformation should return a new object."
    assert mod_em.xy_unit == 'nm'
    assert test_utils.tens_almeq(mod_em.px_size, torch.tensor([100., 100.]))

    assert mod_em.frame_ix.min() == em.frame_ix.min() - 1
    assert test_utils.tens_almeq(mod_em.xyz_nm, em.xyz_nm[:, [1, 0, 2]] * torch.tensor([1., 1., -1.]) + torch.tensor(
        [-150., -50., 0.]))
