import torch
from pytorch_lightning import loggers
from deprecated import deprecated
from typing import Union

from decode import emitter
from decode import simulation
from decode import neuralfitter
from decode import evaluation


def setup_logger(
    cfg,
) -> Union[loggers.LightningLoggerBase, list[loggers.LightningLoggerBase]]:
    """
    Set up logging.

    Args:
        cfg: config
    """
    if cfg["Logging"]["no_op"]:
        return loggers.base.DummyLogger()

    log = []

    if cfg["Logging"]["logger"] == "TensorBoardLogger":
        tb = neuralfitter.logger.TensorboardLogger(save_dir=cfg["Paths"]["logging"])
        log.append(tb)
    else:
        raise NotImplementedError

    return log


def setup_psf(cfg) -> simulation.psf_kernel.PSF:
    from decode import io

    # switch between different psf
    if len(cfg["Simulation"]["PSF"].keys()) >= 2:
        raise NotImplementedError
    if list(cfg["Simulation"]["PSF"].keys())[0] != "CubicSpline":
        raise NotImplementedError

    psf = io.psf.load_spline(
        path=cfg["Paths"]["calibration"],
        xextent=cfg["Simulation"]["psf_extent"]["x"],
        yextent=cfg["Simulation"]["psf_extent"]["y"],
        img_shape=cfg["Simulation"]["img_size"],
        device=cfg["Hardware"]["device"]["simulation"],
        roi_size=cfg["Simulation"]["PSF"]["CubicSpline"]["roi_size"],
        roi_auto_center=cfg["Simulation"]["PSF"]["CubicSpline"]["roi_auto_center"],
    )

    return psf


def setup_background(
    cfg,
) -> tuple[simulation.background.Background, simulation.background.Background]:
    bg_train = simulation.background.BackgroundUniform(
        bg=cfg["Simulation"]["bg_uniform"],
        size=(cfg["Simulation"]["samples"], *cfg["Simulation"]["img_size"]),
        device=cfg["Hardware"]["device"]["simulation"],
    )
    bg_val = simulation.background.BackgroundUniform(
        bg=cfg["Simulation"]["bg_uniform"],
        size=(cfg["Test"]["samples"], *cfg["Simulation"]["img_size"]),
        device=cfg["Hardware"]["device"]["simulation"],
    )
    return bg_train, bg_val


def setup_noise(cfg) -> simulation.camera.CameraEMCCD:
    noise = simulation.camera.CameraEMCCD(
        qe=cfg["Camera"]["qe"],
        spur_noise=cfg["Camera"]["spur_noise"],
        em_gain=cfg["Camera"]["em_gain"],
        e_per_adu=cfg["Camera"]["e_per_adu"],
        baseline=cfg["Camera"]["baseline"],
        read_sigma=cfg["Camera"]["read_sigma"],
        photon_units=cfg["Camera"]["convert2photons"],
        device=cfg["Hardware"]["device"]["simulation"],
    )
    return noise


def setup_structure(cfg) -> simulation.structures.StructurePrior:
    return simulation.structures.RandomStructure(
        xextent=cfg["Simulation"]["emitter_extent"]["x"],
        yextent=cfg["Simulation"]["emitter_extent"]["y"],
        zextent=cfg["Simulation"]["emitter_extent"]["z"],
    )


def setup_code(cfg) -> simulation.code.Code:
    return simulation.code.Code(codes=cfg["Simulation"]["code"])


def setup_model(cfg) -> torch.nn.Module:
    if cfg["Model"]["backbone"] != "SigmaMUNet":
        raise NotImplementedError

    specs = cfg["Model"]["backbone_specs"]
    activation = getattr(torch.nn, cfg["Model"]["backbone_specs"]["activation"])()
    disabled_attr = 3 if cfg["Trainer"]["train_dim"] == 2 else None

    model = neuralfitter.models.SigmaMUNet(
        ch_in=cfg["Model"]["channels_in"],
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
        kaiming_normal=specs.init_custom,
    )
    return model


def setup_loss(cfg) -> neuralfitter.loss.Loss:
    loss = neuralfitter.loss.GaussianMMLoss(
        xextent=cfg["Simulation"]["psf_extent"]["x"],
        yextent=cfg["Simulation"]["psf_extent"]["y"],
        img_shape=cfg["Simulation"]["img_size"],
        device=cfg["Hardware"]["device"]["training"],
        chweight_stat=cfg["Loss"]["ch_weight"],
        reduction="mean",
        return_loggable=True,
    )
    return loss


def setup_optimizer(model: torch.nn.Module, cfg) -> torch.optim.Optimizer:
    catalog = {"Adam": torch.optim.Adam, "AdamW": torch.optim.AdamW}

    opt = catalog[cfg["Optimizer"]["name"]]
    opt = opt(model.parameters(), **cfg["Optimizer"]["specs"])

    return opt


@deprecated(reason="Will be deprecated.", version="0.11")
def setup_scheduler(opt: torch.optim.Optimizer, cfg) -> torch.optim.lr_scheduler.StepLR:
    catalog = {
        "ReduceLROnPlateau": torch.optim.lr_scheduler.ReduceLROnPlateau,
        "StepLR": torch.optim.lr_scheduler.StepLR,
    }
    lr_sched = catalog[cfg["HyperParameter"]["learning_rate_scheduler"]]
    lr_sched = lr_sched(opt, **cfg["HyperParameter"]["learning_rate_scheduler_param"])

    return lr_sched


def setup_em_filter(cfg) -> emitter.process.EmitterProcess:
    if cfg["Target"]["filter"] is not None:
        f = emitter.process.EmitterFilterGeneric(**cfg["Target"]["filter"])
    else:
        f = None

    return f


def setup_frame_scaling(cfg) -> neuralfitter.scale_transform.ScalerAmplitude:
    return neuralfitter.scale_transform.ScalerAmplitude(
        scale=cfg["Scaling"]["input_scale"],
        offset=cfg["Scaling"]["input_offset"],
    )


def setup_tar_scaling(cfg) -> neuralfitter.scale_transform.ScalerTargetList:
    return neuralfitter.scale_transform.ScalerTargetList(
        phot=cfg["Scaling"]["phot_max"],
        z=cfg["Scaling"]["z_max"],
    )


def setup_bg_scaling(cfg) -> neuralfitter.scale_transform.ScalerAmplitude:
    return neuralfitter.scale_transform.ScalerAmplitude(cfg["Scaling"]["bg_max"])


def setup_post_model_scaling(cfg) -> neuralfitter.scale_transform.ScalerModelOutput:
    return neuralfitter.scale_transform.ScalerModelOutput(
        phot=cfg["Scaling"]["phot_max"],
        z=cfg["Scaling"]["z_max"],
        bg=cfg["Scaling"]["z_max"],
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
        xextent=cfg["Simulation"]["frame_extent"]["x"],
        yextent=cfg["Simulation"]["frame_extent"]["y"],
        img_shape=cfg["Simulation"]["img_size"],
    )


def setup_post_process_frame_emitter(
    cfg,
) -> neuralfitter.processing.to_emitter.ToEmitter:
    # last bit that transforms frames to emitters

    if cfg["PostProcessing"]["name"] is None:
        post = neuralfitter.processing.to_emitter.ToEmitterEmpty(
            xy_unit=cfg["Simulation"]["xy_unit"], px_size=cfg["Camera"]["px_size"]
        )

    elif cfg["PostProcessing"]["name"] == "LookUp":
        post = neuralfitter.processing.to_emitter.ToEmitterLookUpPixelwise(
            mask=cfg["PostProcessing"]["specs"]["raw_th"],
            pphotxyzbg_mapping=[0, 1, 2, 3, 4, -1],  # ToDo: remove hard-coding
            xy_unit=cfg["Simulation"]["xy_unit"],
            px_size=cfg["Camera"]["px_size"],
        )

    elif cfg["PostProcessing"]["name"] == "SpatialIntegration":
        post = neuralfitter.processing.to_emitter.ToEmitterSpatialIntegration(
            raw_th=cfg["PostProcessing"]["specs"]["raw_th"],
            xy_unit=cfg["Simulation"]["xy_unit"],
            px_size=cfg["Camera"]["px_size"],
        )
    else:
        raise NotImplementedError

    return post


def setup_matcher(cfg) -> evaluation.match_emittersets.EmitterMatcher:
    matcher = evaluation.match_emittersets.GreedyHungarianMatching(
        match_dims=cfg["Evaluation"]["match_dims"],
        dist_lat=cfg["Evaluation"]["dist_lat"],
        dist_ax=cfg["Evaluation"]["dist_ax"],
        dist_vol=cfg["Evaluation"]["dist_vol"],
    )
    return matcher


def setup_emitter_sampler(
    cfg,
) -> tuple[
    simulation.sampler.EmitterSamplerBlinking, simulation.sampler.EmitterSamplerBlinking
]:
    """
    Get emitter samplers for training and validation set

    Args:
        cfg: config

    Returns:
        - sampler for training set
        - sampler for validation set
    """
    struct = setup_structure(cfg)
    color = setup_code(cfg)

    em_sampler_train = simulation.sampler.EmitterSamplerBlinking(
        structure=struct,
        code=color,
        intensity=(
            cfg["Simulation"]["intensity"]["mean"],
            cfg["Simulation"]["intensity"]["std"]
        ),
        em_num=cfg["Simulation"]["emitter_avg"],
        lifetime=cfg["Simulation"]["lifetime_avg"],
        frame_range=cfg["Simulation"]["samples"],
        xy_unit=cfg["Simulation"]["xy_unit"],
    )

    em_sampler_val = simulation.sampler.EmitterSamplerBlinking(
        structure=struct,
        code=color,
        intensity=(cfg["Simulation"]["intensity"]["mean"], cfg["Simulation"]["intensity"]["std"]),
        em_num=cfg["Simulation"]["emitter_avg"],
        lifetime=cfg["Simulation"]["lifetime_avg"],
        frame_range=cfg["Test"]["samples"],
        xy_unit=cfg["Simulation"]["xy_unit"],
    )

    return em_sampler_train, em_sampler_val


def setup_microscope(
    cfg,
) -> tuple[simulation.microscope.Microscope, simulation.microscope.Microscope]:
    """
    Get microscopes for the training and validation set

    Args:
        cfg: config

    Returns:
        - microscope train set
        - microscope validation set
    """
    psf = setup_psf(cfg)

    raise NotImplementedError("Noise not added.")
    mic_train = simulation.microscope.Microscope(
        psf=psf,
        noise=None,
        frame_range=cfg["Simulation"]["samples"]
    )
    mic_val = simulation.microscope.Microscope(
        psf=psf,
        noise=None,
        frame_range=cfg["Test"]["samples"]
    )

    return mic_train, mic_val


def setup_tar(cfg) -> neuralfitter.target_generator.TargetGenerator:
    scaler = setup_tar_scaling(cfg)
    filter = setup_em_filter(cfg)
    bg_lane = setup_bg_scaling(cfg)

    return neuralfitter.target_generator.TargetGaussianMixture(
        n_max=cfg["Target"]["max_emitters"],
        ix_low=None,
        ix_high=None,
        ignore_ix=True,
        scaler=scaler,
        filter=filter,
        aux_lane=bg_lane,
    )


def setup_processor(cfg):
    scaler_frame = setup_frame_scaling(cfg)
    tar = setup_tar(cfg)
    filter_em = setup_em_filter(cfg)
    post_model = setup_post_model_scaling(cfg)
    post_processor = setup_post_process(cfg)

    return neuralfitter.process.ProcessingSupervised(tar=tar, tar_em=filter_em,
                                                     post_model=post_model,
                                                     post=post_processor)


def setup_sampler(cfg):

    em_train, em_val = setup_emitter_sampler(cfg)
    bg, bg_val = setup_background(cfg)
    proc = setup_processor(cfg)
    mic_train, mic_val = setup_microscope(cfg)

    sampler = neuralfitter.sampler.SamplerSupervised(
        em=em_train,
        bg=bg,
        frames=None,
        proc=proc,
        mic=mic_train,
        bg_mode="sample",
        window=cfg["Trainer"]["frame_window"],
    )

    sampler_val = neuralfitter.sampler.SamplerSupervised(
        em=em_val,
        bg=bg_val,
        frames=None,
        proc=proc,
        mic=mic_val,
        bg_mode="sample",
        window=cfg["Trainer"]["frame_window"],
    )

    return sampler, sampler_val
