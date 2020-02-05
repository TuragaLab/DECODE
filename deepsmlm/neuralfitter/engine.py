import pathlib


class SMLMTrainingEngine:
    """
    This is the main 'training' engine that picks up the data from the simulation engine.
    The convention is, that the first element will be the emitters, the second the camera frames, the third the
    background frames and possible other information (i.e. target data)
    """

    training_engine_folder = pathlib.Path('training_engines')

    def __init__(self, cache_dir, sim_id, em_filter, input_prep, target_gen, weight_gen, engine_id=None):
        """

        Args:
            cache_dir: path to cache directory where the simulation engine stores its simulation data
            sim_id: id of the simulation (engine)
            em_filter: (callable) that filters the emitters as provided by the simulation engine
            input_prep: (callable) that prepares the input data for the network (e.g. rescaling)
            target_gen: (callable) that generates the training data
            weight_gen: (callable) that generates a weight mask corresponding to the target / output data
            engine_id: (optional) set id of this training engine. Otherwise it'll be hostname and time
        """
        self._buffer = []
        self._cache_dir = cache_dir
        self._sim_id = sim_id
        self.em_filter = em_filter
        self.input_prep = input_prep
        self.target_gen = target_gen
        self.weight_gen = weight_gen
        self._engine_id = None

        if engine_id is None:
            self._engine_id = datetime.now().strftime('%b%d_%H-%M-%S') + '_' + socket.gethostname()
        else:
            self._engine_id = engine_id

    def _sign_up_engine(self):
        """
        Signs up this training engine in the experiments folder by placing a text file there.

        """

        """Make sure that sim engine exists."""
        p = pathlib.Path(self._cache_dir) / pathlib.Path(self._sim_id) / self.training_engine_folder
        assert p.exists()
        f = p / pathlib.Path(self._engine_id)

        # place text data there
        with f.open('w+'):
            f.write_text(self._engine_id)

    def _load_and_touch(self):
        """
        Loads the new training data and marks the data as loaded

        Returns:

        """