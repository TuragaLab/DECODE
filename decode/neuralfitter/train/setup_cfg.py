from typing import Optional

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


def setup_trafo_coord(cfg) -> simulation.microscope.XYZTransformationMatrix:
    # ToDo: static, change to real configurable format
    import numpy as np
    from scipy.spatial.transform import Rotation as R

    r = R.from_rotvec(np.pi / 16 * np.array([0, 0, 1]))
    r = torch.from_numpy(r.as_matrix()).float()

    r2 = R.from_rotvec(-0.5 * np.pi / 16 * np.array([0, 0, 1]))
    r2 = torch.from_numpy(r2.as_matrix()).float()

    return simulation.microscope.XYZTransformationMatrix(
        torch.stack([torch.eye(3), r, r2], 0)
    )


def setup_trafo_phot(cfg) -> simulation.microscope.MultiChoricSplitter:
    # ToDo: static, change to real configurable format
    t = torch.tensor([[0.7, 0.3, 0.0], [0.2, 0.7, 0.1], [0.0, 3.0, 0.7]])
    return simulation.microscope.MultiChoricSplitter(t)


def setup_background(
    cfg,
) -> tuple[simulation.background.Background, simulation.background.Background]:
    bg_train = _setup_background_core(cfg, cfg["Simulation"])
    bg_test = _setup_background_core(cfg, cfg["Test"])
    return bg_train, bg_test


def _setup_background_core(cfg, cfg_sim) -> simulation.background.Background:
    return simulation.background.BackgroundUniform(
        bg=cfg_sim["bg"][0]["uniform"],
        size=(cfg_sim["samples"], *cfg_sim["img_size"]),
        device=cfg["Hardware"]["device"]["simulation"],
    )


def setup_cameras(cfg) -> list[simulation.camera.CameraEMCCD]:
    cam = []
    for cfg_cam in cfg["Camera"].values():
        cam.append(
            simulation.camera.CameraEMCCD(
                qe=cfg_cam["qe"],
                spur_noise=cfg_cam["spur_noise"],
                em_gain=cfg_cam["em_gain"],
                e_per_adu=cfg_cam["e_per_adu"],
                baseline=cfg_cam["baseline"],
                read_sigma=cfg_cam["read_sigma"],
                photon_units=cfg_cam["convert2photons"],
                device="cpu",  # cfg["Hardware"]["device"]["simulation"],
            )
        )
    return cam


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
        depth_shared=specs["depth_shared"],
        depth_union=specs["depth_union"],
        initial_features=specs["initial_features"],
        inter_features=specs["inter_features"],
        activation=activation,
        norm=specs["norm"],
        norm_groups=specs["norm_groups"],
        norm_head=specs["norm_head"],
        norm_head_groups=specs["norm_head_groups"],
        pool_mode=specs["pool_mode"],
        upsample_mode=specs["upsample_mode"],
        skip_gn_level=specs["skip_gn_level"],
        disabled_attributes=disabled_attr,
        kaiming_normal=specs["init_custom"],
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
        scale=cfg["Scaling"]["input"]["frame"]["scale"],
        offset=cfg["Scaling"]["input"]["frame"]["offset"],
    )


def setup_aux_scaling(cfg) -> Optional[neuralfitter.scale_transform.ScalerAmplitude]:
    if cfg["Scaling"]["input"]["aux"] is None:
        return None
    return neuralfitter.scale_transform.ScalerAmplitude(
        scale=cfg["Scaling"]["input"]["aux"]["scale"],
        offset=cfg["Scaling"]["input"]["aux"]["offset"],
    )


def setup_input_proc(cfg):
    scaler_frame = setup_frame_scaling(cfg)
    scaler_aux = setup_aux_scaling(cfg)
    cams = setup_cameras(cfg)

    return neuralfitter.processing.model_input.ModelInputPostponed(
        cam=cams,
        scaler_frame=scaler_frame.forward,
        scaler_aux=scaler_aux.forward if scaler_aux is not None else None,
    )


def setup_tar_scaling(cfg) -> neuralfitter.scale_transform.ScalerTargetList:
    return neuralfitter.scale_transform.ScalerTargetList(
        phot=cfg["Scaling"]["output"]["phot"]["max"],
        z=cfg["Scaling"]["output"]["z"]["max"],
    )


def setup_bg_scaling(cfg) -> neuralfitter.scale_transform.ScalerAmplitude:
    return neuralfitter.scale_transform.ScalerAmplitude(
        cfg["Scaling"]["output"]["bg"]["max"]
    )


def setup_post_model_scaling(cfg) -> neuralfitter.scale_transform.ScalerModelOutput:
    return neuralfitter.scale_transform.ScalerModelOutput(
        phot=cfg["Scaling"]["output"]["phot"]["max"],
        z=cfg["Scaling"]["output"]["z"]["max"],
        bg=cfg["Scaling"]["output"]["bg"]["max"],
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
            # ToDo: Generalise
            xy_unit=cfg["Simulation"]["xy_unit"],
            px_size=cfg["Camera"][0]["px_size"],
        )

    elif cfg["PostProcessing"]["name"] == "LookUp":
        post = neuralfitter.processing.to_emitter.ToEmitterLookUpPixelwise(
            mask=cfg["PostProcessing"]["specs"]["raw_th"],
            pphotxyzbg_mapping=[0, 1, 2, 3, 4, -1],  # ToDo: remove hard-coding
            xy_unit=cfg["Simulation"]["xy_unit"],
            px_size=cfg["Camera"][0]["px_size"],
        )

    elif cfg["PostProcessing"]["name"] == "SpatialIntegration":
        post = neuralfitter.processing.to_emitter.ToEmitterSpatialIntegration(
            raw_th=cfg["PostProcessing"]["specs"]["raw_th"],
            xy_unit=cfg["Simulation"]["xy_unit"],
            px_size=cfg["Camera"][0]["px_size"],
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


def setup_evaluator(cfg) -> evaluation.evaluation.EvaluationSMLM:
    matcher = setup_matcher(cfg)
    evaluator = evaluation.evaluation.EvaluationSMLM(
        matcher=matcher,
    )
    return evaluator


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
            cfg["Simulation"]["intensity"]["std"],
        ),
        em_num=cfg["Simulation"]["emitter_avg"],
        lifetime=cfg["Simulation"]["lifetime_avg"],
        frame_range=cfg["Simulation"]["samples"],
        xy_unit=cfg["Simulation"]["xy_unit"],
    )

    em_sampler_val = simulation.sampler.EmitterSamplerBlinking(
        structure=struct,
        code=color,
        intensity=(
            cfg["Simulation"]["intensity"]["mean"],
            cfg["Simulation"]["intensity"]["std"],
        ),
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

    mic_train = simulation.microscope.Microscope(
        psf=psf, noise=None, frame_range=cfg["Simulation"]["samples"]
    )
    mic_val = simulation.microscope.Microscope(
        psf=psf, noise=None, frame_range=cfg["Test"]["samples"]
    )

    return mic_train, mic_val


def _setup_microscope_core(cfg, cfg_sim, psf):
    if cfg_sim.code == [0]:
        m = simulation.microscope.Microscope(
            psf=psf, noise=None, frame_range=cfg_sim["samples"]
        )
    else:
        codes = torch.tensor(cfg_sim.code)
        n_codes = len(codes)
        trafo_xyz = setup_trafo_coord(cfg)
        trafo_phot = setup_trafo_phot(cfg)

        m = simulation.microscope.MicroscopeMultiChannel(
            psf=[psf] * n_codes,
            noise=None,
            trafo_xyz=trafo_xyz,
            trafo_phot=trafo_phot,
            frame_range=cfg_sim["samples"],
            ch_range=(codes.min().item(), codes.max().item() + 1),
        )

        return m


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
    model_input = setup_input_proc(cfg)
    tar = setup_tar(cfg)
    filter_em = setup_em_filter(cfg)
    post_model = setup_post_model_scaling(cfg)
    post_processor = setup_post_process(cfg)

    return neuralfitter.process.ProcessingSupervised(
        m_input=model_input,
        tar=tar,
        tar_em=filter_em,
        post_model=post_model,
        post=post_processor,
    )


def setup_sampler(cfg):

    em_train, em_val = setup_emitter_sampler(cfg)
    bg, bg_val = setup_background(cfg)
    proc = setup_processor(cfg)
    mic_train, mic_val = setup_microscope(cfg)

    sampler = neuralfitter.sampler.SamplerSupervised(
        em=em_train,
        bg=bg,
        frames=None,
        indicator=None,
        proc=proc,
        mic=mic_train,
        bg_mode="sample",
        window=cfg["Trainer"]["frame_window"],
    )

    sampler_val = neuralfitter.sampler.SamplerSupervised(
        em=em_val,
        bg=bg_val,
        frames=None,
        indicator=None,
        proc=proc,
        mic=mic_val,
        bg_mode="sample",
        window=cfg["Trainer"]["frame_window"],
    )

    return sampler, sampler_val
