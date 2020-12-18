import decode.simulation
import decode.utils
import torch

def setup_random_simulation(param):
    """
        Setup the actual simulation

        0. Define PSF function (load the calibration)
        1. Define our struture from which we sample (random prior in 3D) and its photophysics
        2. Define background and noise
        3. Setup simulation and datasets
        """

    psf = decode.utils.calibration_io.SMAPSplineCoefficient(
        calib_file=param.InOut.calibration_file).init_spline(
        xextent=param.Simulation.psf_extent[0],
        yextent=param.Simulation.psf_extent[1],
        img_shape=param.Simulation.img_size,
        cuda_kernel=True if param.Hardware.device_simulation[:4] == 'cuda' else False,
        roi_size=param.Simulation.roi_size,
        roi_auto_center=param.Simulation.roi_auto_center
    )

    """Structure Prior"""
    prior_struct = decode.simulation.structure_prior.RandomStructure.parse(param)

    if param.Simulation.mode in ('acquisition', 'apriori'):
        frame_range_train = (0, param.HyperParameter.pseudo_ds_size)

    elif param.Simulation.mode == 'samples':
        frame_range_train = (-((param.HyperParameter.channels_in - 1) // 2),
                             (param.HyperParameter.channels_in - 1) // 2)
    else:
        raise ValueError

    prior_train = decode.simulation.emitter_generator.EmitterSamplerBlinking.parse(
        param, structure=prior_struct, frames=frame_range_train)

    """Define our background and noise model."""
    bg = decode.simulation.background.UniformBackground.parse(param)

    if param.CameraPreset == 'Perfect':
        noise = decode.simulation.camera.PerfectCamera.parse(param)
    elif param.CameraPreset == 'scmos':
        noise = decode.simulation.camera.SCMOS.parse(param)
#         if param.InOut.cmos_noise:
#             noise_map = torch.load(param.InOut.cmos_noise)
#             if param.InOut.cmos_gain:
#                 noise_map = noise_map / torch.load(param.InOut.cmos_gain)
#             else:
#                 noise_map = noise_map / param.Camera.em_gain
#             noise._read_sigma = noise_map

    elif param.CameraPreset is not None:
        raise NotImplementedError
    else:
        noise = decode.simulation.camera.Photon2Camera.parse(param)

    simulation_train = decode.simulation.simulator.Simulation(psf=psf, em_sampler=prior_train, background=bg, noise=noise, 
                                                              frame_range=frame_range_train)

    frame_range_test = (0, param.TestSet.test_size)

    prior_test = decode.simulation.emitter_generator.EmitterSamplerBlinking.parse(
        param, structure=prior_struct, frames=frame_range_test)

    simulation_test = decode.simulation.simulator.Simulation(psf=psf, em_sampler=prior_test, background=bg, noise=noise,
                                                             frame_range=frame_range_test)

    return simulation_train, simulation_test
