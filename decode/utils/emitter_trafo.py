import torch
from decode.generic import emitter

"""
Predefined transformations, e.g. for the SMLM challenge.
Factors and shifts are with respect to (possibly) changed axis, i.e. in the new system.
"""

challenge_import = {
    'desc': 'Challenge data transformation to match DECODE (this framework) format.',
    'xy_unit': 'nm',
    'px_size': (100., 100.),
    'xyz_axis': (1, 0, 2),
    'xyz_nm_factor': (1., 1., -1.),
    'xyz_nm_shift': (-150., -50., 0.),
    'xyz_px_factor': None,
    'xyz_px_shift': None,
    'frame_ix_shift': -1
}

challenge_export = {
    'desc': 'DECODE (this framework) data transformation to match the challenge format.',
    'xy_unit': 'nm',
    'px_size': (100., 100.),
    'xyz_axis': (1, 0, 2),
    'xyz_nm_factor': (1., 1., -1.),
    'xyz_nm_shift': (50., 150., 0.),
    'xyz_px_factor': None,
    'xyz_px_shift': None,
    'frame_ix_shift': +1
}

deepstorm3d_import = {
    'desc': 'Transformation of DeepStorm output files to be compatible to DECODE.',
    'xy_unit': 'px',
    'px_size': None,
    'xyz_axis': (1, 0, 2),
    'xyz_nm_factor': None,
    'xyz_nm_shift': None,
    'xyz_px_factor': None,
    'xyz_px_shift': None,
    'frame_ix_shift': -1
}


def transform_emitter(em: emitter.EmitterSet, trafo: dict) -> emitter.EmitterSet:
    """
    Transform a set of emitters specified by a transformation dictionary. Returns transformed emitterset.

    Args:
        em: emitterset to be transformed
        trafo: transformation specs

    """

    mod_em = em.clone()

    """Set Px Size"""
    mod_em.px_size = torch.tensor(trafo['px_size']) if trafo['px_size'] is not None else mod_em.px_size

    """Modify proper attributes"""
    if trafo['xyz_axis'] is not None:
        mod_em.xyz = mod_em.xyz[:, trafo['xyz_axis']]
        mod_em.xyz_cr = mod_em.xyz_cr[:, trafo['xyz_axis']]
        mod_em.xyz_sig = mod_em.xyz_sig[:, trafo['xyz_axis']]

    if trafo['xyz_nm_factor'] is not None:
        mod_em.xyz_nm *= torch.tensor(trafo['xyz_nm_factor'])

    if trafo['xyz_nm_shift'] is not None:
        mod_em.xyz_nm += torch.tensor(trafo['xyz_nm_shift'])

    if trafo['xyz_px_factor'] is not None:
        mod_em.xyz_px *= torch.tensor(trafo['xyz_px_factor'])

    if trafo['xyz_px_shift'] is not None:
        mod_em.xyz_px += torch.tensor(trafo['xyz_px_shift'])

    if trafo['frame_ix_shift'] is not None:
        mod_em.frame_ix += torch.tensor(trafo['frame_ix_shift'])

    """Modify unit in which emitter is stored and possibly set px size'"""
    if trafo['xy_unit'] is not None:
        if trafo['xy_unit'] == 'nm':
            mod_em.xyz_nm = mod_em.xyz_nm
        elif trafo['xy_unit'] == 'px':
            mod_em.xyz_px = mod_em.xyz_px
        else:
            raise ValueError(f"Unsupported unit ({trafo['xy_unit']}).")

    return mod_em
