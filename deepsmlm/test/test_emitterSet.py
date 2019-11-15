import torch
from unittest import TestCase
import pytest
import time
import os

from deepsmlm.generic.emitter import EmitterSet, RandomEmitterSet, EmptyEmitterSet
import deepsmlm.generic.emitter as emitter
import deepsmlm.test.utils_ci as tutil


class TestEmitterSet:

    @pytest.fixture(scope='class')
    def em2d(self):
        return EmitterSet(xyz=torch.rand((25, 2)),
                               phot=torch.rand(25),
                               frame_ix=torch.zeros(25))

    @pytest.fixture(scope='class')
    def em3d(self):
        frames = torch.arange(25)
        frames[[0, 1, 2]] = 1
        return EmitterSet(xyz=torch.rand((25, 3)),
                               phot=torch.rand(25),
                               frame_ix=frames)

    def test_init(self, em2d, em3d):
        # 2D input get's converted to 3D with zeros
        assert em2d.xyz.shape[1] == 3
        assert em3d.xyz.shape[1] == 3

    def test_split_in_frames(self, em2d, em3d):
        splits = em2d.split_in_frames(None, None)
        assert splits.__len__() == 1

        splits = em3d.split_in_frames(None, None)
        assert em3d.frame_ix.max() - em3d.frame_ix.min() + 1 == splits.__len__()

        """Test negative numbers in Frame ix."""
        neg_frames = EmitterSet(torch.rand((3, 3)),
                                torch.rand(3),
                                torch.tensor([-1, 0., 1]))
        splits = neg_frames.split_in_frames(None, None)
        assert  splits.__len__() == 3
        splits = neg_frames.split_in_frames(0, None)
        assert splits.__len__() == 2

    def test_adjacent_frame_split(self):
        xyz = torch.rand((500, 3))
        phot = torch.rand_like(xyz[:, 0])
        frame_ix = torch.randint_like(xyz[:, 0], low=-1, high=2)
        em = EmitterSet(xyz, phot, frame_ix)

        em_split = em.split_in_frames(-1, 1)
        assert (em_split[0].frame_ix == -1).all()
        assert (em_split[1].frame_ix == 0).all()
        assert (em_split[2].frame_ix == 1).all()

        em_split = em.split_in_frames(0, 0)
        assert em_split.__len__() == 1
        assert (em_split[0].frame_ix == 0).all()

        em_split = em.split_in_frames(-1, -1)
        assert em_split.__len__() == 1
        assert (em_split[0].frame_ix == -1).all()

        em_split = em.split_in_frames(1, 1)
        assert em_split.__len__() == 1
        assert (em_split[0].frame_ix == 1).all()

    def test_cat_emittersets(self):
        """
        Test the concatenation of two emittersets.
        :return:
        """
        sets = [RandomEmitterSet(50), RandomEmitterSet(20)]
        cat_sets = EmitterSet.cat_emittersets(sets, None, 1)
        assert 70 == cat_sets.num_emitter
        assert 0 == cat_sets.frame_ix[0]
        assert 1 == cat_sets.frame_ix[50]

        sets = [RandomEmitterSet(50), RandomEmitterSet(20)]
        cat_sets = EmitterSet.cat_emittersets(sets, torch.tensor([5, 50]), None)
        assert 70 == cat_sets.num_emitter
        assert 5 == cat_sets.frame_ix[0]
        assert 50 == cat_sets.frame_ix[50]

    def test_sanity_check(self):

        """Test correct shape of 1D tensors in EmitterSet"""
        xyz = torch.rand((10, 3))
        phot = torch.rand((10, 1))
        frame_ix = torch.rand(10)
        with pytest.raises(ValueError):
            EmitterSet(xyz, phot, frame_ix)

        """Test correct number of el. in EmitterSet."""
        xyz = torch.rand((10, 3))
        phot = torch.rand((11, 1))
        frame_ix = torch.rand(10)
        with pytest.raises(ValueError):
            EmitterSet(xyz, phot, frame_ix)


    def test_write_to_csv(self):
        """
        Test to write to csv file.
        :return:
        """
        deepsmlm_root = os.path.abspath(
            os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         os.pardir, os.pardir)) + '/'

        random_em = RandomEmitterSet(1000)
        fname = deepsmlm_root + 'deepsmlm/test/assets/dummy_csv.txt'
        random_em.write_to_csv(fname)
        assert os.path.isfile(fname)
        os.remove(fname)

    def test_write_to_SMAP(self):
        deepsmlm_root = os.path.abspath(
            os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         os.pardir, os.pardir)) + '/'

        random_em = RandomEmitterSet(1000)
        fname = deepsmlm_root + 'deepsmlm/test/assets/dummy_csv.txt'
        random_em.write_csv_smap(fname)
        assert os.path.isfile(fname)
        # os.remove(fname)

    def test_eq(self):
        em = RandomEmitterSet(1000)
        em2 = em.clone()

        assert em == em2


def test_empty_emitterset():
    em = EmptyEmitterSet()
    assert 0 == em.num_emitter


class TestLooseEmitterSet:

    @pytest.fixture(scope='class')
    def dummy_set(self):
        num_emitters = 10000
        t0_max = 5000
        em = emitter.LooseEmitterSet(torch.rand((num_emitters, 3)),
                                torch.ones(num_emitters) * 10000,
                                None,
                                torch.rand(num_emitters) * t0_max,
                                torch.rand(num_emitters) * 3)

        return em

    def test_distribution(self):
        loose_em = emitter.LooseEmitterSet(torch.zeros((2, 3)),
                                           torch.tensor([1000., 10]),
                                           None,
                                           torch.tensor([-0.2, 0.9]),
                                           torch.tensor([1., 5]))

        em = loose_em.return_emitterset()
        print("Done")

    @pytest.mark.skip(reason="C++ currently not implemented for updated generator.")
    def test_eq_distribute(self, dummy_set):
        """
        Test whether the C++ and the Python implementations return the same stuff.
        """
        # Time the stuff
        t = time.process_time()
        xyz_py, phot_py, frame_py, id_py = dummy_set.distribute_framewise_py()
        t_py = time.process_time() - t
        t = time.process_time()
        xyz_cpp, phot_cpp, frame_cpp, id_cpp = dummy_set.distribute_framewise_cpp()
        t_cpp = time.process_time() - t

        assert tutil.tens_almeq(xyz_py, xyz_cpp)
        assert tutil.tens_almeq(phot_py, phot_cpp)
        assert tutil.tens_almeq(frame_py, frame_cpp)
        assert tutil.tens_almeq(id_py, id_cpp)

        """Print timing."""
        print("Elapsed time Python: {} - C++: {}".format(t_py, t_cpp))
