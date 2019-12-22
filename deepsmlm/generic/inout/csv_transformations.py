import torch

"""Predefined Transformations."""
pre_trafo = {
    'challenge': {
        'comments': 'Challenge format export.',
        'xyz_shift': [150., 50., 0.],
        'xy_unit': 'nm',
        'frame_shift': 1,
        'axis': [1, 0, 2],
        'plain_header': True
    },
    'smap': {
        'comments': 'SMAP format export.',
        'xyz_shift': None,
        'xy_unit': None,
        'frame_shift': None,
        'axis': [1, 0, 2],
        'plain_header': True
    }
}

