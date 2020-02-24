import torch
import torch.nn
import numpy as np

import deepsmlm.generic.emitter
from scipy.stats import gaussian_kde


def density_estimate(emitter, band_with=0.18):
    """
    Estimates the local density of emitters.
    Args:
        emitter: (EmitterSet)
        band_with: (float) default 0.18, should be in or lower than magnitude of px size in px coordinates

    Returns: emitterset with density estimate as bg value

    """

    em_split = emitter.split_in_frames(0)

    for em_ in em_split:
        xyc = em_.xyz_px[:, :2]
        xy = np.vstack([xyc[:, 0], xyc[:, 1]])

        d = gaussian_kde(xy, bw_method=band_with)(xy)
        d = torch.from_numpy(d)
        em_.bg = d.float()

    return deepsmlm.generic.emitter.EmitterSet.cat(em_split)
