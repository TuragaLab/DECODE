import pytest
import torch

from decode.neuralfitter import coord_transform


def test_offset2coordinate():
    p = coord_transform.Offset2Coordinate(
        xextent=(-0.5, 3.5),
        yextent=(1., 2.),
        img_shape=(4, 2),
    )

    offsets = torch.tensor([
        [  # offsets x
            [-1, 2, 0, 0.5],
            [0., 9.]
        ],
        [  # offsets y
            [0., 2, 3, 2],
            [9., 8.],
        ],

    ])
    expct = torch.tensor([
        [  # x absolute
            [-1, 3, 2, 3.5],
            [0., 12.],
        ],
        [  # y absolute
            [1.25, 3.25, 4.25, 3.25],
            [9.75, 8.75]
        ]

    ])

    out = p.forward(offsets)



