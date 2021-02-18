from typing import Tuple, Optional

from ... import neuralfitter


def get_preprocessing(scale_offset: Tuple[float, ...], camera: Optional = None,
                      auto_crop: Optional[int] = None, mirror_dim: Optional[int] = None):
    """
    Default frame processing pipeline.

    Args:
        scale_offset: tuple containg amplitude rescaling plus offset
        camera: must be parsed if frames should be converted to photon units and must not if not
        auto_crop: crop frame to size that is multiple of a pixel fold
        mirror_dim: mirror a specific dimension (useful for experimental data)

    """

    proc_sequence = []

    if camera is not None:
        proc_sequence.append(neuralfitter.utils.processing.wrap_callable(camera.backward))

    if auto_crop is not None:
        proc_sequence.append(neuralfitter.frame_processing.AutoCenterCrop(auto_crop))

    if mirror_dim is not None:
        proc_sequence.append(neuralfitter.frame_processing.Mirror2D(dims=mirror_dim))

    proc_sequence.append(neuralfitter.scale_transform.AmplitudeRescale(*scale_offset))
    proc_sequence = neuralfitter.utils.processing.TransformSequence(proc_sequence)

    return proc_sequence


def get_postprocessing():
    return