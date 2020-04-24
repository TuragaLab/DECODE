import datetime
import hashlib
import socket
import pathlib
import pickle
import time


class SMLMTrainingEngine:
    """
    This is the main 'training' engine that picks up the data from the simulation engine.
    The convention is, that the first element will be the emitters, the second the camera frames, the third the
    background frames and possible other information (i.e. target data)
    """

    training_engine_folder = pathlib.Path('training_engines')

    def __init__(self, cache_dir, sim_id, engine_id=None, test_set=False):
        """

        Args:
            cache_dir: path to cache directory where the simulation engine stores its simulation data
            sim_id: id of the simulation (engine)
            engine_id: (optional) set id of this training engine. Otherwise it'll be hostname and time
            test_set (bool): is engine for test set
        """
        self._buffer = []
        self._touched_data = []
        self._cache_dir = cache_dir
        self._sim_id = sim_id
        self._engine_id = None
        self.test_set = test_set

        if engine_id is None:
            self._engine_id = datetime.datetime.now().strftime('%b%d_%H-%M-%S') + '_' + socket.gethostname()
        else:
            self._engine_id = engine_id

        self._sign_up_engine()

    def _sign_up_engine(self):
        """
        Signs up this training engine in the experiments folder by placing a text file there.

        """

        """Make sure that sim engine exists."""
        p = pathlib.Path(self._cache_dir) / pathlib.Path(self._sim_id) / self.training_engine_folder
        if not p.exists():
            raise RuntimeError(f"Simulation engine does not exist, or at least not in the specified path."
                               f"\nPath:{str(p)}")
        f = p / pathlib.Path(self._engine_id + '.txt')

        # place text data there
        with f.open('w+'):
            f.write_text(self._engine_id)

    def _update_buffer(self):
        """
        Maintains a list of simulation data which has not yet been loaded.
        Goes through the folder every time

        Returns:

        """

        self._buffer = []

        """Get list of subfolders named traindata in experiment"""
        p = pathlib.Path(self._cache_dir) / pathlib.Path(self._sim_id)
        assert p.exists()

        if not self.test_set:
            # get all subfolders that start with traindata
            self._buffer = [str(x.name) for x in p.iterdir() if (x.is_dir()) and (str(x.name)[:9] == 'traindata')]

        else:
            self._buffer = [str(x.name) for x in p.iterdir() if (x.is_dir()) and (str(x.name) == 'testdata')]

        # list of incomplete buffer elements which can not yet be loaded (i.e. folder is present, but file is not yet)
        _incomplete_elements = [buff_el for buff_el in self._buffer if not (p / buff_el / buff_el).is_file()]

        # rm all elements that have been seen by this engine
        self._buffer = list(set(self._buffer).difference(self._touched_data, _incomplete_elements))

        self._buffer.sort()

    def _check_wait_buffer(self):
        """
        Checks the buffer and waits when its empty

        Returns:

        """
        while True:
            self._update_buffer()
            if len(self._buffer) == 0:
                print("Waiting for training data ...", end="\r")
                time.sleep(5)
            else:
                break

    def load_and_touch(self):
        """
        Loads the new training data and marks the data as loaded

        Returns:

        """

        self._check_wait_buffer()
        bel = self._buffer.pop(0)
        bel_folder_path = pathlib.Path(self._cache_dir) / pathlib.Path(self._sim_id) / pathlib.Path(bel)
        bel_path = bel_folder_path / pathlib.Path(bel)

        """Load training data."""
        assert bel_path.is_file(), "Dataset file not present."

        with bel_path.open('rb') as f:
            data = f.read()

        bel_path_hash_file = pathlib.Path(str(bel_path) + '_hash.txt')
        with bel_path_hash_file.open('r') as f:
            data_hash = f.read()

        # Compare hashes and if they do not match, recurse function
        if data_hash != hashlib.sha256(data).hexdigest():
            print("Validation hash did not agree with the one next to the data to be loaded. "
                  "Maybe file write is in progress? Waiting ...")
            self.load_and_touch()

        data = pickle.loads(data)

        """Leave touch"""
        touch_path = bel_folder_path / pathlib.Path(self._engine_id + '.txt')
        with touch_path.open('w+') as f:
            f.write(self._engine_id)

        self._touched_data.append(str(bel_folder_path.name))

        print(f"Dataset {str(bel_folder_path.name)} loaded and marked.")

        return data
