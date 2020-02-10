"""
This module sets up a simulation engine which writes to a binary
"""
import copy
import math
import itertools
import torch

import deepsmlm.neuralfitter.utils.pytorch_customs

torch.multiprocessing.set_sharing_strategy('file_system')
import torch.utils
import tqdm
import time
import pickle
from pathlib import Path

import deepsmlm.generic.emitter
import deepsmlm.generic.utils.data_utils as deepsmlm_utils


class SimulationEngine:
    """
    Simulation engine.
    Note that the elements of the datasets must be pickable!
    """
    def __init__(self, cache_dir, exp_id, cpu_worker, buffer_size, ds_train, ds_test=None):

        self.cache_dir = cache_dir
        self.exp_id = exp_id
        self.exp_dir = None
        self.cpu_worker = cpu_worker
        self._batch_size = None
        self.buffer = []
        self.buffer_size = buffer_size
        self._train_engines = []
        self._train_engines_path = None
        self._train_data_ix = -1

        self.ds_train = ds_train
        self.ds_test = ds_test

        self._dl_train = None
        self._dl_test = None

        self.setup_dataloader()
        self._setup_exchange()

        print("Simulation engine setup up done.")

    def _setup_exchange(self):
        """
        Sets up the exchange folder of the simulation engine after some sanity checking.

        """

        # check existence of exchange folder
        cache_path = Path(self.cache_dir)
        assert cache_path.is_dir()

        # create subfolder with id as specified
        exp_path = Path(self.cache_dir) / Path(self.exp_id)
        exp_path.mkdir()
        self.exp_dir = exp_path

        # creates folder in which training engines sign up
        eng_path = self.exp_dir / 'training_engines'
        eng_path.mkdir()
        self._train_engines_path = eng_path

    def setup_dataloader(self, batch_size=256):
        """
        Sets up the dataloader for the simulation engine.

        Args:
            batch_size: (int) the batch_size for the dataloaders to produce the training samples. High values reduce
            thread overhead, but can lead to shared memory issues.

        Returns:

        """
        """If the ds is small, reduce the batch_size accordingly to utilise all workers."""
        if len(self.ds_train) < batch_size * self.cpu_worker:
            batch_size = math.ceil(len(self.ds_train) / batch_size)

        self._batch_size = batch_size

        self._dl_train = torch.utils.data.DataLoader(dataset=self.ds_train, batch_size=self._batch_size, shuffle=False,
                                                     num_workers=self.cpu_worker, collate_fn=deepsmlm.neuralfitter.utils.pytorch_customs.smlm_collate,
                                                     pin_memory=False)

        if self.ds_test is not None:
            if len(self.ds_test) < batch_size * self.cpu_worker:
                batch_size = math.ceil(len(self.ds_test) / batch_size)

            batch_size_test = batch_size

            self._dl_test = torch.utils.data.DataLoader(dataset=self.ds_test, batch_size=batch_size_test, shuffle=False,
                                                        num_workers=self.cpu_worker, collate_fn=deepsmlm.neuralfitter.utils.pytorch_customs.smlm_collate,
                                                        pin_memory=False)

    @staticmethod
    def _get_engines(folderpath):
        """
        Checks active training engines by checking .txt files.
        """
        if not isinstance(folderpath, Path):
            folderpath = Path(folderpath)

        """Get list of all training engines"""
        engines_txt = folderpath.glob('*.txt')
        engines = [eng.stem for eng in engines_txt]
        engines.sort()

        return engines

    def _get_train_engines(self):
        """
        Gets the training engines in the folder where they are maintained

        """

        self._train_engines = self._get_engines(self._train_engines_path)

    def _relax_buffer(self):
        """
        Checks for each element in the buffer whether all engines have loaded the data already.

        """

        # update engines
        self._get_train_engines()

        # go through elements in buffer
        for i, bel in enumerate(self.buffer):
            bel_fpath = self.exp_dir / bel
            # get engines that loaded this data already
            eng_loaded = self._get_engines(bel_fpath)

            # remove buffer element when all active engines saw this data already
            # if (self._train_engines == eng_loaded) and (len(self._train_engines) >= 1):
            if set(self._train_engines).issubset(eng_loaded) and (len(self._train_engines) >= 1):
                deepsmlm_utils.del_dir(bel_fpath, False)
                del self.buffer[i]

        print('Buffer checked and cleared. Waiting for training engines to pick up the data.', end="\r")

    @staticmethod
    def reduce_batch_dim(batches):
        """
        Transforms from list of batches with content elements into collection of content elements

        Args:
            batches:

        Returns:

        """
        # n_batches = len(batches)
        sample_instance = [x[0][0] for x in batches[0]]
        n_types = len(sample_instance)

        con = [None] * n_types

        for i, t in enumerate(sample_instance):
            if isinstance(t, deepsmlm.generic.emitter.EmitterSet):
                con[i] = [e[i] for e in batches]  # list of lists of emittersets
                con[i] = list(itertools.chain.from_iterable(con[i]))
            elif isinstance(t, torch.Tensor):
                con[i] = torch.cat([e[i] for e in batches], dim=0)
            else:
                raise NotImplementedError("Unsupported type for reduction.")

        return con

    @staticmethod
    def run_pickle_dl(dl, folder, filename):
        """
        Runs the dataloader for a single epoch and pickles the results into a file
        Args:
            dl: dataloader
            folder: folder in which the data should be placed in
            filename: output filename

        Returns:

        """
        dl_out = []
        for dl_batch in tqdm.tqdm(dl):
            dl_batch_ = copy.deepcopy(dl_batch)  # otherwise you get multiprocessing issues
            del dl_batch
            dl_out.append(dl_batch_)

        # if batch size was more than 1, collate
        dl_out = SimulationEngine.reduce_batch_dim(dl_out)

        out_folder = Path(folder)
        assert not out_folder.exists()
        out_folder.mkdir()

        file = folder / filename
        with open(str(file), 'wb+') as f:
            pickle.dump(dl_out, f, protocol=-1)

    def run(self, n_max=None):
        """
        Main method to run the simulation engine.
        Simulates test data once if not None; simulates epochs of training data until buffer is full; clears cached
        training data if loaded by all active training engines;

        Args:
            n_max: (integer, None) maximum number of loops. Rarely needed (but needed for testing)

        Returns:

        """

        """Write test data once."""
        if self._dl_test is not None:
            self.run_pickle_dl(self._dl_test, self.exp_dir / 'testdata', 'testdata')
            print("Finished computation of test data.")

        """Check if buffer is full, otherwise simulate"""
        n = 0
        while n_max is None or n < n_max:
            # check buffer
            if len(self.buffer) >= self.buffer_size:
                # possibly clear buffer element if all engines touched
                self._relax_buffer()
                time.sleep(5)  # add some rest here because if the engine needs to wait it's kept in the loop

            else:
                # generate new training data
                self._train_data_ix += 1
                train_data_name = ('traindata_' + str(self._train_data_ix))
                self.buffer.append(train_data_name)
                self.run_pickle_dl(self._dl_train, self.exp_dir / train_data_name, train_data_name)

            n += 1
