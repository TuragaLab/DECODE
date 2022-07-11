import torch
from pytorch_lightning import loggers

from decode import emitter
from decode import simulation
from decode import neuralfitter
from decode import evaluation


def setup_logger(cfg) -> list[loggers.LightningLoggerBase]:
    """
    Set up logging.

    Args:
        cfg: config
    """
    if cfg.Logging.no_op:
        return [loggers.base.DummyLogger()]

    return [getattr(loggers, l)(**cfg.Logging.logger[l]) for l in cfg.Logging.logger]


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
        xextent=cfg.Simulation.psf_extent[0],
        yextent=cfg.Simulation.psf_extent[1],
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
        raise NotImplementedError("Automatic camera chose not yet implemented.")
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


def setup_structure(cfg) -> simulation.structures.StructurePrior:
    return simulation.structures.RandomStructure(
        xextent=cfg.Simulation.emitter_extent[0],
        yextent=cfg.Simulation.emitter_extent[1],
        zextent=cfg.Simulation.emitter_extent[2],
    )


def setup_model(cfg) -> torch.nn.Module:
    catalog = {
        "SigmaMUNet": neuralfitter.models.SigmaMUNet,
        "DoubleMUnet": neuralfitter.models.model_param.DoubleMUnet,
        "SimpleSMLMNet": neuralfitter.models.model_param.SimpleSMLMNet,
    }

    model = catalog[cfg.HyperParameter.architecture]
    # ToDo: Get rid .parse?
    return model.parse(cfg)


def setup_loss(cfg) -> neuralfitter.loss.Loss:
    loss = neuralfitter.loss.GaussianMMLoss(
        xextent=cfg.Simulation.psf_extent[0],
        yextent=cfg.Simulation.psf_extent[1],
        img_shape=cfg.Simulation.img_size,
        device=cfg.Hardware.device,
        chweight_stat=cfg.HyperParameter.chweight_stat,
    )
    return loss


def setup_optimizer(model: torch.nn.Module, cfg) -> torch.optim.Optimizer:
    catalog = {"Adam": torch.optim.Adam, "AdamW": torch.optim.AdamW}

    opt = catalog[cfg.HyperParameter.optimizer]
    opt = opt(model.parameters(), **cfg.HyperParameter.opt_param)

    return opt


def setup_scheduler(opt: torch.optim.Optimizer, cfg) -> torch.optim.lr_scheduler.StepLR:
    catalog = {
        "ReduceLROnPlateau": torch.optim.lr_scheduler.ReduceLROnPlateau,
        "StepLR": torch.optim.lr_scheduler.StepLR,
    }
    lr_sched = catalog[cfg.HyperParameter.learning_rate_scheduler]
    lr_sched = lr_sched(opt, **cfg.HyperParameter.learning_rate_scheduler_param)

    return lr_sched


def setup_em_filter(cfg) -> emitter.process.EmitterProcess:
    if cfg.HyperParameter.emitter_label_photon_min is not None:
        f = emitter.process.PhotonFilter(cfg.HyperParameter.emitter_label_photon_min)
    else:
        f = emitter.process.EmitterIdentity()

    return f


def setup_tar(ix_low, ix_high, cfg):
    return neuralfitter.utils.processing.TransformSequence(
        [
            setup_tar_tensor_parameter(ix_low=ix_low, ix_high=ix_high, cfg=cfg),
            setup_tar_disable(cfg),
            setup_tar_scaling(cfg),
        ]
    )


def setup_tar_tensor_parameter(
    ix_low, ix_high, cfg
) -> neuralfitter.target_generator.ParameterListTarget:
    return neuralfitter.target_generator.ParameterListTarget(
        n_max=cfg.HyperParameter.max_number_targets,
        xextent=cfg.Simulation.psf_extent[0],
        yextent=cfg.Simulation.psf_extent[1],
        ix_low=ix_low,
        ix_high=ix_high,
        squeeze_batch_dim=True,
    )


def setup_tar_disable(cfg) -> neuralfitter.target_generator.DisableAttributes:
    return neuralfitter.target_generator.DisableAttributes(
        attr_ix=cfg.HyperParameter.disabled_attributes
    )


def setup_tar_scaling(cfg) -> neuralfitter.scale_transform.ParameterListRescale:
    return neuralfitter.scale_transform.ParameterListRescale(
        phot_max=cfg.Scaling.phot_max,
        z_max=cfg.Scaling.z_max,
        bg_max=cfg.Scaling.bg_max,
    )


def setup_post_process(cfg) -> neuralfitter.post_processing.PostProcessing:
    post_frame_em = setup_post_process_frame_emitter(cfg)

    post = neuralfitter.utils.processing.TransformSequence(
        [
            neuralfitter.scale_transform.InverseParamListRescale(
                phot_max=cfg.Scaling.phot_max,
                z_max=cfg.Scaling.z_max,
                bg_max=cfg.Scaling.bg_max,
            ),
            neuralfitter.coord_transform.Offset2Coordinate.parse(cfg),
            post_frame_em,
        ]
    )
    return post


def setup_post_process_frame_emitter(
    cfg,
) -> neuralfitter.post_processing.PostProcessing:
    # last bit that transforms frames to emitters

    if cfg.PostProcessing is None:
        post = neuralfitter.post_processing.NoPostProcessing(
            xy_unit="px", px_size=cfg.Camera.px_size
        )

    elif cfg.PostProcessing == "LookUp":
        post = neuralfitter.post_processing.LookUpPostProcessing(
            raw_th=cfg.PostProcessingParam.raw_th,
            pphotxyzbg_mapping=[0, 1, 2, 3, 4, -1],  # ToDo: remove hard-coding
            xy_unit="px",
            px_size=cfg.Camera.px_size,
        )

    elif cfg.PostProcessing == "SpatialIntegration":
        post = neuralfitter.post_processing.SpatialIntegration(
            raw_th=cfg.PostProcessingParam.raw_th,
            xy_unit="px",
            px_size=cfg.Camera.px_size,
        )
    else:
        raise NotImplementedError

    return post


def setup_matcher(cfg) -> evaluation.match_emittersets.EmitterMatcher:
    matcher = evaluation.match_emittersets.GreedyHungarianMatching(
        match_dims=cfg.Evaluation.match_dims,
        dist_lat=cfg.Evaluation.dist_lat,
        dist_ax=cfg.Evaluation.dist_ax,
        dist_vol=cfg.Evaluation.dist_vol,
    )
    return matcher
