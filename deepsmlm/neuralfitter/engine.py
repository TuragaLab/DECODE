import datetime
import socket
import pathlib
import pickle
import time

import deepsmlm.simulation.engine


class SMLMTrainingEngine:
    """
    This is the main 'training' engine that picks up the data from the simulation engine.
    The convention is, that the first element will be the emitters, the second the camera frames, the third the
    background frames and possible other information (i.e. target data)
    """

    training_engine_folder = pathlib.Path('training_engines')

    def __init__(self, cache_dir, sim_id, engine_id=None):
        """

        Args:
            cache_dir: path to cache directory where the simulation engine stores its simulation data
            sim_id: id of the simulation (engine)
            engine_id: (optional) set id of this training engine. Otherwise it'll be hostname and time
        """
        self._buffer = []
        self._touched_data = []
        self._cache_dir = cache_dir
        self._sim_id = sim_id
        self._engine_id = None

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
        assert p.exists()
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

        # get all subfolders that start with traindata
        self._buffer = [str(x.name) for x in p.iterdir() if (x.is_dir()) and (str(x.name)[:9] == 'traindata')]

        # rm all elements that have been seen by this engine
        self._buffer = list(set(self._buffer).difference(self._touched_data))
        
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
        assert bel_path.is_file()
        with bel_path.open('rb') as f:
            data = pickle.load(f)

        """Leave touch"""
        touch_path = bel_folder_path / pathlib.Path(self._engine_id + '.txt')
        with touch_path.open('w+') as f:
            f.write(self._engine_id)

        self._touched_data.append(str(bel_folder_path.name))

        print(f"Training data {str(bel_folder_path.name)} loaded and touched.")

        return data


if __name__ == '__main__':
    import os
    deepsmlm_root = os.path.abspath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     os.pardir, os.pardir)) + '/'

    engine = SMLMTrainingEngine(cache_dir=deepsmlm_root + 'deepsmlm/test/assets/sim_engine',
                                sim_id='dummy_data')

    while True:
        dat = engine.load_and_touch()
