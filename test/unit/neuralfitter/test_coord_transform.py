import torch

from decode.neuralfitter import coord_transform


def test_offset2coordinate():
    p = coord_transform.Offset2Coordinate(
        xextent=(-0.5, 2.5),
        yextent=(1.0, 2.0),
        img_shape=(3, 2),
    )

    offsets = torch.tensor(
        [
            [  # x offsets
                [1, 2],
                [3, 4],
                [5, 6],
            ],
            [  # y offsets
                [-0.5, -1.0],
                [-1.5, -2.0],
                [-2.5, -3.0],
            ],
        ]
    )

    expct = torch.tensor(
        [
            [  # x coord
                [1, 2],
                [4, 5],
                [7, 8],
            ],
            [  # y coord
                [0.75, 0.75],
                [-0.25, -0.25],
                [-1.25, -1.25],
            ],
        ]
    )

    out = p.forward(offsets.unsqueeze(0))

    assert (out == expct.unsqueeze(0)).all()
