import numpy as np
import torch

from decode.emitter.emitter import factory
from decode.utils import emitter_trafo


def test_emitter_transformation():
    em = factory(200, extent=10000, xy_unit="px", px_size=(100.0, 100.0))
    em.frame_ix = torch.randint_like(em.frame_ix, 1, 10000)

    mod_em = emitter_trafo.transform_emitter(em, emitter_trafo.challenge_import)

    assert id(em) != id(mod_em), "Emitter transformation should return a new object."
    assert not mod_em.eq_meta(em)

    assert mod_em.frame_ix.min() == em.frame_ix.min() - 1
    np.testing.assert_array_equal(
        mod_em.xyz_nm,
        em.xyz_nm[:, [1, 0, 2]] * torch.tensor([1.0, 1.0, -1.0])
        + torch.tensor([-150.0, -50.0, 0.0]),
    )
