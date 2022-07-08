from pytorch_lightning import loggers


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


def setup_psf(cfg):
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


def setup_background():
    pass


def setup_noise():
    pass


def setup_prior():
    pass
