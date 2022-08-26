
def autoset_scaling(param):
    def set_if_none(var, value):
        if var is None:
            var = value
        return var

    param.Scaling.input_scale = set_if_none(param.Scaling.input_scale, param.Simulation.intensity_mu_sig[0] / 50)
    param.Scaling.phot_max = set_if_none(param.Scaling.phot_max,
                                         param.Simulation.intensity_mu_sig[0] + 8 * param.Simulation.intensity_mu_sig[
                                             1])

    param.Scaling.z_max = set_if_none(param.Scaling.z_max, param.Simulation.emitter_extent[2][1] * 1.2)
    if param.Scaling.input_offset is None:
        if isinstance(param.Simulation.bg_uniform, (list, tuple)):
            param.Scaling.input_offset = (param.Simulation.bg_uniform[1] + param.Simulation.bg_uniform[0]) / 2
        else:
            param.Scaling.input_offset = param.Simulation.bg_uniform

    if param.Scaling.bg_max is None:
        if isinstance(param.Simulation.bg_uniform, (list, tuple)):
            param.Scaling.bg_max = param.Simulation.bg_uniform[1] * 1.2
        else:
            param.Scaling.bg_max = param.Simulation.bg_uniform * 1.2

    return param