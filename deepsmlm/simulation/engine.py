import copy
import hashlib
import itertools
import math
from abc import ABC, abstractmethod

import torch

torch.multiprocessing.set_sharing_strategy('file_system')
from torch.utils.data import Dataset

import deepsmlm.neuralfitter.utils.pytorch_customs

import torch.utils
import tqdm
import time
import pickle
from pathlib import Path

import deepsmlm.generic.emitter
import deepsmlm.simulation
import deepsmlm.generic.utils.data_utils as deepsmlm_utils


class SimulationEngine(ABC):

    def __init__(self, cache_dir, exp_id, buffer_size, write_testdata):
        super().__init__()

        self.cache_dir = cache_dir
        self.exp_id = exp_id
        self.exp_dir = None
        self.write_testdata = write_testdata

        self._train_engines = []
        self._train_engines_path = None
        self._train_data_ix = -1

        self.buffer = []
        self.buffer_size = buffer_size

        self._setup_exchange()

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
    def save(dataset, folder, filename):

        out_folder = Path(folder)
        assert not out_folder.exists()
        out_folder.mkdir()

        # monitor time to disk
        t0 = time.time()
        file = folder / filename

        pickle_out = pickle.dumps(dataset, protocol=-1)
        # write hash first, then the actual file
        hash_file = Path(str(file) + '_hash').with_suffix('.txt')

        with hash_file.open('w+') as f:
            f.write(hashlib.sha256(pickle_out).hexdigest())

        with file.open('wb+') as f:
            f.write(pickle_out)

        t1 = time.time()
        print(f"Wrote dataset {filename} in {t1 - t0:.2f}s to disk. Filesize: {file.stat().st_size / 10 ** 6:.1f} MB "
              f"(i.e. {(file.stat().st_size / 10 ** 6) / (t1 - t0):.1f} MB/s)")

    @abstractmethod
    def sample_train(self):
        raise NotImplementedError

    @abstractmethod
    def sample_test(self):
        raise NotImplementedError

    def run(self, n_max: int = None):

        """Write testdata once"""
        if self.write_testdata:
            testdata = self.sample_test()
            self.save(testdata, self.exp_dir / 'testdata', 'testdata')

        """Check if buffer is full, otherwise simulate."""
        n = 0
        while n_max is None or n < n_max:

            """Check and Clear Buffer"""
            if len(self.buffer) >= self.buffer_size:
                # possibly clear buffer element if all engines touched
                self._relax_buffer()
                time.sleep(5)  # add some rest here because if the engine needs to wait it's kept in the loop

            else:  # generate new training samples and save the,
                self._train_data_ix += 1
                train_data_name = ('traindata_' + str(self._train_data_ix))
                self.buffer.append(train_data_name)

                data_train = self.sample_train()
                self.save(data_train, self.exp_dir / train_data_name, train_data_name)

            n += 1


class DatasetStreamEngine(SimulationEngine):

    def __init__(self, cache_dir: (str, Path), exp_id: str, buffer_size: int,
                 sim_train: deepsmlm.simulation.Simulation, sim_test: deepsmlm.simulation.Simulation = None):
        super().__init__(cache_dir=cache_dir, exp_id=exp_id, buffer_size=buffer_size,
                         write_testdata=True if sim_test is not None else False)

        self.sim_train = sim_train
        self.sim_test = sim_test

        self.sanity_check()

    def sanity_check(self):
        assert self.sim_train.em_sampler is not None, "Simulation Engine must contain emitter sampler."

    @staticmethod
    def _sample_data(simulator):
        frames, bg_frames, emitter = simulator.forward()
        # emitter = emitter.split_in_frames(0, len(frames) - 1)

        return frames.cpu(), bg_frames.cpu(), emitter

    def sample_train(self):
        return self._sample_data(self.sim_train)

    def sample_test(self):
        return self._sample_data(self.sim_test)


class SampleStreamEngine(SimulationEngine):
    """
    Simulation engine.
    Note that the elements of the datasets must be pickable!
    """

    def __init__(self, cache_dir: (str, Path), exp_id: str, cpu_worker: int, buffer_size: int, ds_train, ds_test=None):

        super().__init__(cache_dir=cache_dir, exp_id=exp_id, buffer_size=buffer_size,
                         write_testdata=True if ds_test is not None else False)

        self.cpu_worker = cpu_worker
        self._batch_size = None

        self.ds_train = ds_train
        self.ds_test = ds_test

        self._dl_train = None
        self._dl_test = None

        self._setup_dataloader()

        print("Simulation engine setup up done.")

    def _setup_dataloader(self, batch_size=16):
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
                                                     num_workers=self.cpu_worker,
                                                     collate_fn=deepsmlm.neuralfitter.utils.pytorch_customs.smlm_collate,
                                                     pin_memory=False)

        if self.ds_test is not None:
            if len(self.ds_test) < batch_size * self.cpu_worker:
                batch_size = math.ceil(len(self.ds_test) / batch_size)

            batch_size_test = batch_size

            self._dl_test = torch.utils.data.DataLoader(dataset=self.ds_test, batch_size=batch_size_test, shuffle=False,
                                                        num_workers=self.cpu_worker,
                                                        collate_fn=deepsmlm.neuralfitter.utils.pytorch_customs.smlm_collate,
                                                        pin_memory=False)

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

    @classmethod
    def sample_data(cls, dl):

        dl_out = []
        for dl_batch in tqdm.tqdm(dl):
            dl_batch_ = copy.deepcopy(dl_batch)  # otherwise you get multiprocessing issues
            del dl_batch
            dl_out.append(dl_batch_)

        # if batch size was more than 1, collate
        dl_out = cls.reduce_batch_dim(dl_out)

        return dl_out

    def sample_train(self):
        return self.sample_data(self._dl_train)

    def sample_test(self):
        return self.sample_data(self._dl_test)


class SMLMSimulationDatasetOnFly(Dataset):
    """
    Simple implementation of a dataset which can generate samples from a simulator and returns them along with the
    emitters that are on the frame.
    I did this mainly here because I did not want to care about the multiprocessing myself and rather use the pytorch
    dataset thingy which does that for me.
    In itself this class will not be used for training a network directly.
    """

    def __init__(self, simulator, ds_size: int):
        """

        Args:
            simulator: (Simulation) (in principle anything with a) forward method
            ds_size: (int) size of the dataset
        """
        super().__init__()
        self.sim = simulator
        self.ds_size = ds_size

        # make sure that simulator has a prior to sample from and not a static emitter set
        assert not isinstance(self.sim.em, deepsmlm.generic.emitter.EmitterSet)

    def __len__(self):
        return self.ds_size

    def __getitem__(self, item):
        """
        Returns the items

        Args:
            item: (int) index of sample

        Returns:
            em_tar: emitter target
            cam_frames: camera frames
            bg_frames: background frames
        """

        cam_frames, bg_frames, em_tar = self.sim.forward()
        return em_tar, cam_frames, bg_frames
