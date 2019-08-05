import json


def write_params(filename, inout, logging, scheduling, hyper, simulation, camera, scaling, postprocessing, evaluation):
    """
    Writes a JSON file with the used parameters.
    :param filename:
    :param inout:
    :param logging:
    :param scheduling:
    :param hyper:
    :param simulation:
    :param camera:
    :param scaling:
    :param postprocessing:
    :param evaluation:
    :return:
    """
    json_string = json.dumps({'InOutParameter': inout._asdict(),
                              'LoggerParameter': logging._asdict(),
                              'HyperParameter': hyper._asdict(),
                              'SchedulerParameter': scheduling._asdict(),
                              'SimulationParam': simulation._asdict(),
                              'CameraParam': camera._asdict(),
                              'ScalingParam': scaling._asdict(),
                              'PostProcessingParam': postprocessing._asdict(),
                              'EvaluationParam': evaluation._asdict()}, indent=4)

    f = open(filename, "w+")
    f.write(json_string)
    f.close()


def load_params(filename):
    pass