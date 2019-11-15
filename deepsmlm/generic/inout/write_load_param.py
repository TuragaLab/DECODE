import json
import warnings


def write_params(filename, param):
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
    data = {'InOut': param['InOut'],
            'Hardware': param['Hardware'],
            'Logging': param['Logging'],
            'HyperParameter': param['HyperParameter'],
            'LearningRateScheduler': param['LearningRateScheduler'],
            'SimulationScheduler': param['SimulationScheduler'],
            'Simulation': param['Simulation'],
            'Camera': param['Camera'],
            'Scaling': param['Scaling'],
            'PostProcessing': param['PostProcessing'],
            'Evaluation': param['Evaluation']}

    with open(filename, "w") as write_file:
        json.dump(data, write_file, indent=4)


def load_params(filename):
    with open(filename) as json_file:
        params = json.load(json_file)

    return params
