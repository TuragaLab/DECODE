import torch
from pytorch_lightning import loggers
from deprecated import deprecated
from typing import Union

from decode import emitter
from decode import simulation
from decode import neuralfitter
from decode import evaluation


def setup_logger(cfg) -> Union[loggers.LightningLoggerBase, list[loggers.LightningLoggerBase]]:
    """
    Set up logging.

    Args:
        cfg: config
    """
    if cfg.Logging.no_op:
        return loggers.base.DummyLogger()

    l = []

    if "TensorBoardLogger" in cfg.Logging.logger:
        if (kwargs := cfg.Logging.logger.TensorBoardLogger) is None:
            kwargs = dict()
        tb = loggers.TensorBoardLogger(save_dir=cfg.Paths.logging, **kwargs)
        l.append(tb)
    else:
        raise NotImplementedError

    return l


def setup_psf(cfg) -> simulation.psf_kernel.PSF:
    from decode import io

    # switch between different psf
    if len(cfg.Simulation.PSF) >= 2:
        raise NotImplementedError
    if list(cfg.Simulation.PSF.keys())[0] != "CubicSpline":
        raise NotImplementedError

    psf = io.psf.load_spline(
        path=cfg.Paths.calibration,
        xextent=cfg.Simulation.psf_extent.x,
        yextent=cfg.Simulation.psf_extent.y,
        img_shape=cfg.Simulation.img_size,
        device=cfg.Hardware.device_simulation,
        roi_size=cfg.Simulation.PSF.CubicSpline.roi_size,
        roi_auto_center=cfg.Simulation.PSF.CubicSpline.roi_auto_center,
    )

    return psf


def setup_background(cfg) -> simulation.background.Background:
    return simulation.background.BackgroundUniform(cfg.Simulation.bg_uniform)


def setup_noise(cfg) -> simulation.camera.Camera:
    if cfg.CameraPreset == "Perfect":
        noise = simulation.camera.CameraPerfect(device=cfg.Hardware.device_simulation)
    elif cfg.CameraPreset is not None:
        raise NotImplementedError("Automatic camera chose not yet implemented.")
    else:
        noise = simulation.camera.CameraEMCCD(
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
        xextent=cfg.Simulation.emitter_extent.x,
        yextent=cfg.Simulation.emitter_extent.y,
        zextent=cfg.Simulation.emitter_extent.z,
    )


def setup_code(cfg) -> simulation.code.Code:
    return simulation.code.Code(codes=cfg.Simulation.code)


def setup_model(cfg) -> torch.nn.Module:
    if cfg.Model.backbone != "SigmaMUNet":
        raise NotImplementedError

    specs = cfg.Model.backbone_specs
    activation = getattr(torch.nn, cfg.Model.backbone_specs.activation)()
    disabled_attr = 3 if cfg.Trainer.train_dim == 2 else None

    model = neuralfitter.models.SigmaMUNet(
        ch_in=cfg.Model.channels_in,
        depth_shared=specs.depth_shared,
        depth_union=specs.depth_union,
        initial_features=specs.initial_features,
        inter_features=specs.inter_features,
        activation=activation,
        norm=specs.norm,
        norm_groups=specs.norm_groups,
        norm_head=specs.norm_head,
        norm_head_groups=specs.norm_head_groups,
        pool_mode=specs.pool_mode,
        upsample_mode=specs.upsample_mode,
        skip_gn_level=specs.skip_gn_level,
        disabled_attributes=disabled_attr,
        kaiming_normal=specs.init_custom
    )
    return model


def setup_loss(cfg) -> neuralfitter.loss.Loss:
    loss = neuralfitter.loss.GaussianMMLoss(
        xextent=cfg.Simulation.psf_extent.x,
        yextent=cfg.Simulation.psf_extent.y,
        img_shape=cfg.Simulation.img_size,
        device=cfg.Hardware.device,
        chweight_stat=cfg.Loss.ch_weight,
    )
    return loss


def setup_optimizer(model: torch.nn.Module, cfg) -> torch.optim.Optimizer:
    catalog = {"Adam": torch.optim.Adam, "AdamW": torch.optim.AdamW}

    opt = catalog[cfg.Optimizer.name]
    opt = opt(model.parameters(), **cfg.Optimizer.specs)

    return opt


@deprecated(reason="Will be deprecated.", version="0.11")
def setup_scheduler(opt: torch.optim.Optimizer, cfg) -> torch.optim.lr_scheduler.StepLR:
    catalog = {
        "ReduceLROnPlateau": torch.optim.lr_scheduler.ReduceLROnPlateau,
        "StepLR": torch.optim.lr_scheduler.StepLR,
    }
    lr_sched = catalog[cfg.HyperParameter.learning_rate_scheduler]
    lr_sched = lr_sched(opt, **cfg.HyperParameter.learning_rate_scheduler_param)

    return lr_sched


def setup_em_filter(cfg) -> emitter.process.EmitterProcess:
    if cfg.Target.filter is not None:
        f = emitter.process.EmitterFilterGeneric(**cfg.Target.filter)
    else:
        f = None

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
) -> neuralfitter.target_generator.ParameterList:
    return neuralfitter.target_generator.ParameterList(
        n_max=cfg.Target.max_emitters,
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


def setup_tar_scaling(cfg) -> neuralfitter.scale_transform.ScalerTargetList:
    return neuralfitter.scale_transform.ScalerTargetList(
        phot=cfg.Scaling.phot_max,
        z=cfg.Scaling.z_max,
        bg_max=cfg.Scaling.bg_max,
    )


def setup_post_process(cfg) -> neuralfitter.processing.post.PostProcessing:
    post = neuralfitter.processing.post.PostProcessingGaussianMixture(
        scaler=setup_post_model_scaling(cfg),
        coord_convert=setup_post_process_offset(cfg),
        frame_to_emitter=setup_post_process_frame_emitter(cfg),
    )
    return post


def setup_post_process_offset(cfg) -> neuralfitter.coord_transform.Offset2Coordinate:
    return neuralfitter.coord_transform.Offset2Coordinate(
        xextent=cfg.Simulation.frame_extent.x,
        yextent=cfg.Simulation.frame_extent.y,
        img_shape=cfg.Simulation.img_size,
    )


def setup_post_process_frame_emitter(
    cfg,
) -> neuralfitter.processing.to_emitter.ToEmitter:
    # last bit that transforms frames to emitters

    if cfg.PostProcessing.name is None:
        post = neuralfitter.processing.to_emitter.ToEmitterEmpty(
            xy_unit=cfg.Simulation.xy_unit,
            px_size=cfg.Camera.px_size
        )

    elif cfg.PostProcessing.name == "LookUp":
        post = neuralfitter.processing.to_emitter.ToEmitterLookUpPixelwise(
            mask=cfg.PostProcessing.specs.raw_th,
            pphotxyzbg_mapping=[0, 1, 2, 3, 4, -1],  # ToDo: remove hard-coding
            xy_unit=cfg.Simulation.xy_unit,
            px_size=cfg.Camera.px_size,
        )

    elif cfg.PostProcessing.name == "SpatialIntegration":
        post = neuralfitter.processing.to_emitter.ToEmitterSpatialIntegration(
            raw_th=cfg.PostProcessing.specs.raw_th,
            xy_unit=cfg.Simulation.xy_unit,
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
