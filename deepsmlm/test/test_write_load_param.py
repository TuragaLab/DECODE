import os
import pytest
import torch

import deepsmlm.neuralfitter.arguments as param
import deepsmlm.generic.inout.write_load_param as wlp

"""Root folder"""
deepsmlm_root = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)),
                 os.pardir, os.pardir)) + '/'


def test_write_param():
    filename = deepsmlm_root + 'deepsmlm/test/assets/test_param.json'

    """Set ur basic parameters"""
    io_par = param.InOutParameter(
        root=deepsmlm_root,
        log_comment='',
        data_mode='online',
        data_set=None,  # deepsmlm_root + 'data/2019-03-26/complete_z_range.npz',
        model_out=deepsmlm_root + 'network/2019-08-02/model_challenge_re.pt',
        model_init=deepsmlm_root + 'network/2019-08-02/model_challenge_7.pt')

    log_par = param.LoggerParameter(
        tags=['3D', 'Offset', 'UNet'])

    sched_par = param.SchedulerParameter(
        lr_factor=0.1,
        lr_patience=10,
        lr_threshold=0.0025,
        lr_cooldown=10,
        lr_verbose=True,
        sim_factor=1,
        sim_patience=1,
        sim_threshold=0,
        sim_cooldown=10,
        sim_verbose=True,
        sim_disabled=True,
        sim_max_value=50,
    )

    hy_par = param.HyperParameter(
        dimensions=3,
        channels=3,
        max_emitters=64,
        min_phot=0.,
        data_lifetime=10,
        upscaling=None,
        upscaling_mode=None,
        batch_size=32,
        test_size=256,
        num_epochs=1000,
        lr=1E-4,
        device=str(torch.device('cuda')),
        ignore_boundary_frames=True,
        speiser_weight_sqrt_phot=False,
        class_freq_weight=None,
        pch_weight=1.  # 2 * 20 / (64*64)
    )

    sim_par = param.SimulationParam(
        pseudo_data_size=(512 * hy_par.batch_size + hy_par.test_size),  # (128*32 + 128),
        emitter_extent=((-0.5, 63.5), (-0.5, 63.5), (-750, 750)),
        psf_extent=((-0.5, 63.5), (-0.5, 63.5), (-750., 750.)),
        img_size=(64, 64),
        density=0,
        emitter_av=20,
        photon_range=None,
        bg_pois=95,
        calibration=deepsmlm_root +
                    'data/calibration/2019-06-13_Calibration/sequence-as-stack-Beads-AS-Exp_3dcal.mat',
        intensity_mu_sig=(10000., 500.),
        lifetime_avg=2.
    )

    cam_par_challenge = param.CameraParam(
        qe=1.,
        em_gain=300.,
        e_per_adu=45.,
        baseline=100.,
        read_sigma=74.4,
        spur_noise=0.002
    )

    scale_par = param.ScalingParam(
        dx_max=0.6,
        dy_max=0.6,
        z_max=750.,
        phot_max=50000.,
        linearisation_buffer=1.2
    )

    post_par = param.PostProcessingParam(
        single_val_th=0.2,
        total_th=0.5
    )

    eval_par = param.EvaluationParam(
        dist_lat=1.5,
        dist_ax=300,
        match_dims=2
    )

    wlp.write_params(filename, io_par, log_par, sched_par, hy_par, sim_par, cam_par_challenge, scale_par, post_par, eval_par)
    exists = os.path.isfile(filename)
    assert exists
    os.remove(filename)