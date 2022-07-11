from pytorch_lightning import loggers

from decode import simulation


def setup_logger(cfg) -> list[loggers.LightningLoggerBase]:
    """
    Set up logging.

    Args:
        cfg: config
    """
    if cfg.Logging.no_op:
        return [loggers.base.DummyLogger()]

    return [getattr(loggers, l)(**cfg.Logging.logger[l]) for l in cfg.Logging.logger]


def setup_processing(cfg):
    pass


def setup_psf(cfg) -> simulation.psf_kernel.PSF:
    from decode import io

    # switch between different psf
    if len(cfg.Simulation.PSF) >= 2:
        raise NotImplementedError
    if list(cfg.Simulation.PSF.keys())[0] != "CubicSpline":
        raise NotImplementedError

    psf = io.psf.SMAPSplineCoefficient(
        calib_file=cfg.InOut.calibration_file
    ).init_spline(
        xextent=cfg.Simulation.PSF.CubicSpline.psf_extent[0],
        yextent=cfg.Simulation.PSF.CubicSpline.psf_extent[1],
        img_shape=cfg.Simulation.img_size,
        device=cfg.Hardware.device_simulation,
        roi_size=cfg.Simulation.PSF.CubicSpline.roi_size,
        roi_auto_center=cfg.Simulation.PSF.CubicSpline.roi_auto_center,
    )

    return psf


def setup_background(cfg) -> simulation.background.Background:
    return simulation.background.UniformBackground(cfg.Simulation.bg_uniform)


def setup_noise(cfg) -> simulation.camera.Camera:
    if cfg.CameraPreset == "Perfect":
        noise = simulation.camera.PerfectCamera(device=cfg.Hardware.device_simulation)
    elif cfg.CameraPreset is not None:
        raise NotImplementedError("Automatic camera chose not yet impleted.")
    else:
        noise = simulation.camera.Photon2Camera(
            qe=cfg.Camera.qe,
            spur_noise=cfg.Camera.spur_noise,
            em_gain=cfg.Camera.em_gain,
            e_per_adu=cfg.Camera.e_per_adu,
            baseline=cfg.Camera.baseline,
            read_sigma=cfg.Camera.read_sigma,
            photon_units=cfg.Camera.convert2photons,
            device=cfg.Hardware.device_simulation,
        )
    return noise


def setup_prior(cfg) -> simulation.structures.StructurePrior:
    return simulation.structures.RandomStructure(
        xextent=cfg.Simulation.emitter_extent[0],
        yextent=cfg.Simulation.emitter_extent[1],
        zextent=cfg.Simulation.emitter_extent[2],
    )
