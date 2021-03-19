from pathlib import Path
from unittest import mock

import numpy as np
import pytest
import torch

import decode.generic.emitter as emitter
from decode.generic import test_utils
from decode.generic.emitter import EmitterSet, CoordinateOnlyEmitter, RandomEmitterSet, EmptyEmitterSet


@pytest.fixture()
def em2d():
    """Effectively 2D EmitterSet"""

    return EmitterSet(xyz=torch.rand((25, 2)),
                      phot=torch.rand(25),
                      frame_ix=torch.zeros(25, dtype=torch.long))


@pytest.fixture()
def em3d():
    """Most basic (i.e. all necessary fields) 3D EmitterSet"""

    frames = torch.arange(25, dtype=torch.long)
    frames[[0, 1, 2]] = 1
    return EmitterSet(xyz=torch.rand((25, 3)),
                      phot=torch.rand(25),
                      frame_ix=frames)


@pytest.fixture
def em3d_full(em3d):
    return EmitterSet(xyz=em3d.xyz,
                      phot=em3d.phot,
                      bg=torch.rand_like(em3d.phot) * 100,
                      frame_ix=em3d.frame_ix,
                      id=em3d.id,
                      xyz_sig=torch.rand_like(em3d.xyz),
                      phot_sig=torch.rand_like(em3d.phot) * em3d.phot.sqrt(),
                      xyz_cr=torch.rand_like(em3d.xyz) ** 2,
                      phot_cr=torch.rand_like(em3d.phot) * em3d.phot.sqrt() * 1.5,
                      bg_cr=torch.rand_like(em3d.phot),
                      xy_unit='nm',
                      px_size=(100., 200.))


class TestEmitterSet:

    def test_properties(self, em2d, em3d, em3d_full):

        for em in (em2d, em3d, em3d_full):
            em.phot_scr
            em.bg_scr

            if em.px_size is not None and em.xy_unit is not None:
                em.xyz_px
                em.xyz_nm
                em.xyz_scr
                em.xyz_scr_px
                em.xyz_scr_nm
                em.xyz_sig_px
                em.xyz_sig_nm
                em.xyz_sig_tot_nm
                em.xyz_sig_weighted_tot_nm

        # ToDo: Test auto conversion

    def test_dim(self, em2d, em3d):

        assert em2d.dim() == 2
        assert em3d.dim() == 3

    def test_xyz_shape(self, em2d, em3d):
        """
        Tests shape and correct data type
        Args:
            em2d: fixture (see above)
            em3d: fixture (see above)

        Returns:

        """

        # 2D input get's converted to 3D with zeros
        assert em2d.xyz.shape[1] == 3
        assert em3d.xyz.shape[1] == 3

        assert em3d.frame_ix.dtype in (torch.int, torch.long, torch.short)

    xyz_conversion_data = [  # xyz_input, # xy_unit, #px-size # expect px, # expect nm
        (torch.empty((0, 3)), None, None, "err", "err"),
        (torch.empty((0, 3)), 'px', None, torch.empty((0, 3)), "err"),
        (torch.empty((0, 3)), 'nm', None, "err", torch.empty((0, 3))),
        (torch.tensor([[25., 25., 5.]]), None, None, "err", "err"),
        (torch.tensor([[25., 25., 5.]]), 'px', None, torch.tensor([[25., 25., 5.]]), "err"),
        (torch.tensor([[25., 25., 5.]]), 'nm', None, "err", torch.tensor([[25., 25., 5.]])),
        (torch.tensor([[.25, .25, 5.]]), 'px', (50., 100.), torch.tensor([[.25, .25, 5.]]),
         torch.tensor([[12.5, 25., 5.]])),
        (torch.tensor([[25., 25., 5.]]), 'nm', (50., 100.), torch.tensor([[.5, .25, 5.]]),
         torch.tensor([[25., 25., 5.]]))
    ]

    @pytest.mark.parametrize("xyz_input,xy_unit,px_size,expct_px,expct_nm", xyz_conversion_data)
    @pytest.mark.filterwarnings("ignore:UserWarning")
    def test_xyz_conversion(self, xyz_input, xy_unit, px_size, expct_px, expct_nm):

        """Init and expect warning if specified"""
        em = emitter.CoordinateOnlyEmitter(xyz_input, xy_unit=xy_unit, px_size=px_size)

        """Test the respective units"""
        if isinstance(expct_px, str) and expct_px == "err":
            with pytest.raises(ValueError):
                _ = em.xyz_px
        else:
            assert test_utils.tens_almeq(em.xyz_px, expct_px)

        if isinstance(expct_nm, str) and expct_nm == "err":
            with pytest.raises(ValueError):
                _ = em.xyz_nm

        else:
            assert test_utils.tens_almeq(em.xyz_nm, expct_nm)

    xyz_cr_conversion_data = [  # xyz_scr_input, # xy_unit, #px-size # expect_scr_px, # expect scr_nm
        (torch.empty((0, 3)), None, None, "err", "err"),
        (torch.empty((0, 3)), 'px', None, torch.empty((0, 3)), "err"),
        (torch.empty((0, 3)), 'nm', None, "err", torch.empty((0, 3))),
        (torch.tensor([[25., 25., 5.]]), None, None, "err", "err"),
        (torch.tensor([[25., 25., 5.]]), 'px', None, torch.tensor([[25., 25., 5.]]), "err"),
        (torch.tensor([[25., 25., 5.]]), 'nm', None, "err", torch.tensor([[25., 25., 5.]])),
        (torch.tensor([[.25, .25, 5.]]), 'px', (50., 100.), torch.tensor([[.25, .25, 5.]]),
         torch.tensor([[12.5, 25., 5.]])),
        (torch.tensor([[25., 25., 5.]]), 'nm', (50., 100.), torch.tensor([[.5, .25, 5.]]),
         torch.tensor([[25., 25., 5.]]))
    ]

    @pytest.mark.parametrize("xyz_scr_input,xy_unit,px_size,expct_px,expct_nm", xyz_cr_conversion_data)
    @pytest.mark.filterwarnings("ignore:UserWarning")
    def test_xyz_cr_conversion(self, xyz_scr_input, xy_unit, px_size, expct_px, expct_nm):
        """
        Here we test the cramer rao unit conversion. We can reuse the testdata as for the xyz conversion because it does
        not make a difference for the test candidate.

        """

        """Init and expect warning if specified"""
        em = emitter.CoordinateOnlyEmitter(torch.rand_like(xyz_scr_input), xy_unit=xy_unit, px_size=px_size)
        em.xyz_cr = xyz_scr_input ** 2

        """Test the respective units"""
        if isinstance(expct_px, str) and expct_px == "err":
            with pytest.raises(ValueError):
                _ = em.xyz_cr_px
        else:
            assert test_utils.tens_almeq(em.xyz_scr_px, expct_px)

        if isinstance(expct_nm, str) and expct_nm == "err":
            with pytest.raises(ValueError):
                _ = em.xyz_cr_nm

        else:
            assert test_utils.tens_almeq(em.xyz_scr_nm, expct_nm)

    @pytest.mark.parametrize("attr,power", [('xyz', 1),
                                            ('xyz_sig', 1),
                                            ('xyz_cr', 2)])
    def test_property_conversion(self, attr, power, em3d_full):
        with mock.patch.object(emitter.EmitterSet, '_pxnm_conversion') as conversion:
            getattr(em3d_full, attr + '_nm')

        conversion.assert_called_once_with(getattr(em3d_full, attr), in_unit='nm', tar_unit='nm', power=power)

    @mock.patch.object(emitter.EmitterSet, 'cat')
    def test_add(self, mock_add):
        em_0 = emitter.RandomEmitterSet(20)
        em_1 = emitter.RandomEmitterSet(100)

        _ = em_0 + em_1
        mock_add.assert_called_once_with((em_0, em_1), None, None)

    def test_iadd(self):
        em_0 = emitter.RandomEmitterSet(20)
        em_1 = emitter.RandomEmitterSet(50)

        em_0 += em_1
        assert len(em_0) == 70

    def test_chunk(self):

        big_em = RandomEmitterSet(100000)

        splits = big_em.chunks(10000)
        re_merged = EmitterSet.cat(splits)

        assert sum([len(e) for e in splits]) == len(big_em)
        assert re_merged == big_em

        # test not evenly splittable number
        em = RandomEmitterSet(7)
        splits = em.chunks(3)

        assert len(splits[0]) == 3
        assert len(splits[1]) == 2
        assert len(splits[-1]) == 2

    def test_split_in_frames(self, em2d, em3d):
        splits = em2d.split_in_frames(None, None)
        assert splits.__len__() == 1

        splits = em3d.split_in_frames(None, None)
        assert em3d.frame_ix.max() - em3d.frame_ix.min() + 1 == len(splits)

        """Test negative numbers in Frame ix."""
        neg_frames = EmitterSet(torch.rand((3, 3)),
                                torch.rand(3),
                                torch.tensor([-1, 0, 1]))
        splits = neg_frames.split_in_frames(None, None)
        assert splits.__len__() == 3
        splits = neg_frames.split_in_frames(0, None)
        assert splits.__len__() == 2

    def test_adjacent_frame_split(self):
        xyz = torch.rand((500, 3))
        phot = torch.rand_like(xyz[:, 0])
        frame_ix = torch.randint_like(xyz[:, 0], low=-1, high=2).int()
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

        sets = [RandomEmitterSet(50), RandomEmitterSet(20)]
        cat_sets = EmitterSet.cat(sets, None, 1)
        assert 70 == len(cat_sets)
        assert 0 == cat_sets.frame_ix[0]
        assert 1 == cat_sets.frame_ix[50]

        sets = [RandomEmitterSet(50), RandomEmitterSet(20)]
        cat_sets = EmitterSet.cat(sets, torch.tensor([5, 50]), None)
        assert 70 == len(cat_sets)
        assert 5 == cat_sets.frame_ix[0]
        assert 50 == cat_sets.frame_ix[50]

        # test correctness of px size and xy unit
        sets = [RandomEmitterSet(50, xy_unit='px', px_size=(100., 200.)), RandomEmitterSet(20)]
        em = EmitterSet.cat(sets)
        assert em.xy_unit == 'px'
        assert (em.px_size == torch.tensor([100., 200.])).all()

    def test_split_cat(self):
        """
        Tests whether split and cat (and sort by ID) returns the same result as the original starting.

        """

        em = RandomEmitterSet(1000)
        em.id = torch.arange(len(em))
        em.frame_ix = torch.randint_like(em.frame_ix, 10000)

        """Run"""
        em_split = em.split_in_frames(0, 9999)
        em_re_merged = EmitterSet.cat(em_split)

        """Assertions"""
        # sort both by id
        ix = torch.argsort(em.id)
        ix_re = torch.argsort(em_re_merged.id)

        assert em[ix] == em_re_merged[ix_re]

    @pytest.mark.parametrize("frac", [0., 0.1, 0.5, 0.9, 1.])
    def test_sigma_filter(self, frac):

        """Setup"""
        em = emitter.RandomEmitterSet(10000)
        em.xyz_sig = (torch.randn_like(em.xyz_sig) + 5).clamp(0.)

        """Run"""
        out = em.filter_by_sigma(fraction=frac)

        """Assert"""
        assert len(em) * frac == pytest.approx(len(out))

    def test_hist_detection(self):

        em = emitter.RandomEmitterSet(10000)
        em.prob = torch.rand_like(em.prob)
        em.xyz_sig = torch.randn_like(em.xyz_sig) * torch.tensor([1., 2., 3.]).unsqueeze(0)

        """Run"""
        out = em.hist_detection()

        """Assert"""
        assert set(out.keys()) == {'prob', 'sigma_x', 'sigma_y', 'sigma_z'}

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

    @pytest.mark.parametrize("em", [emitter.RandomEmitterSet(25, 64, px_size=(100., 125.)),
                                    emitter.EmptyEmitterSet(xy_unit='nm', px_size=(100., 125.))])
    def test_inplace_replace(self, em):
        em_start = emitter.RandomEmitterSet(25, xy_unit='px', px_size=None)
        em_start._inplace_replace(em)

        assert em_start == em

    @pytest.mark.parametrize("format", ['.pt', '.h5', '.csv'])
    @pytest.mark.filterwarnings("ignore:.*For .csv files, implicit usage of .load()")
    def test_save_load(self, format, tmpdir):

        em = RandomEmitterSet(1000, xy_unit='nm', px_size=(100., 100.))

        p = Path(tmpdir / f'em{format}')
        em.save(p)
        em_load = EmitterSet.load(p)
        assert em == em_load, "Reloaded emitterset is not equivalent to inital one."

    @pytest.mark.parametrize("em_a,em_b,expct", [(CoordinateOnlyEmitter(torch.tensor([[0., 1., 2.]])),
                                                  CoordinateOnlyEmitter(torch.tensor([[0., 1., 2.]])),
                                                  True),
                                                 (CoordinateOnlyEmitter(torch.tensor([[0., 1., 2.]]), xy_unit='px'),
                                                  CoordinateOnlyEmitter(torch.tensor([[0., 1., 2.]]), xy_unit='nm'),
                                                  False),
                                                 (CoordinateOnlyEmitter(torch.tensor([[0., 1., 2.]]), xy_unit='px'),
                                                  CoordinateOnlyEmitter(torch.tensor([[0., 1.1, 2.]]), xy_unit='px'),
                                                  False)
                                                 ])
    def test_eq(self, em_a, em_b, expct):

        if expct:
            assert em_a == em_b
        else:
            assert not (em_a == em_b)

    def test_meta(self):

        em = RandomEmitterSet(100, xy_unit='nm', px_size=(100., 200.))
        assert set(em.meta.keys()) == {'xy_unit', 'px_size'}

    def test_data(self):
        return  # implicitly in test_to_dict

    def test_to_dict(self):

        em = RandomEmitterSet(100, xy_unit='nm', px_size=(100., 200.))

        """Check whether doing one round of to_dict and back works"""
        em_clone = em.clone()

        em_dict = EmitterSet(**em.to_dict())
        assert em_clone == em_dict


def test_empty_emitterset():
    em = EmptyEmitterSet()
    assert 0 == len(em)


class TestLooseEmitterSet:

    def test_sanity(self):
        with pytest.raises(ValueError) as err:  # wrong xyz dimension
            _ = emitter.LooseEmitterSet(xyz=torch.rand((20, 1)), intensity=torch.ones(20),
                                        ontime=torch.ones(20), t0=torch.rand(20), id=None, xy_unit='px', px_size=None)
        assert str(err.value) == "Wrong xyz dimension."

        with pytest.raises(ValueError) as err:  # non unique IDs
            _ = emitter.LooseEmitterSet(xyz=torch.rand((20, 3)), intensity=torch.ones(20),
                                        ontime=torch.ones(20), t0=torch.rand(20), id=torch.ones(20), xy_unit='px',
                                        px_size=None)
        assert str(err.value) == "IDs are not unique."

        with pytest.raises(ValueError) as err:  # negative intensity
            _ = emitter.LooseEmitterSet(xyz=torch.rand((20, 3)), intensity=-torch.ones(20),
                                        ontime=torch.ones(20), t0=torch.rand(20), id=None, xy_unit='px', px_size=None)
        assert str(err.value) == "Negative intensity values encountered."

        with pytest.raises(ValueError) as err:  # negative intensity
            _ = emitter.LooseEmitterSet(xyz=torch.rand((20, 3)), intensity=torch.ones(20),
                                        ontime=-torch.ones(20), t0=torch.rand(20), id=None, xy_unit='px', px_size=None)
        assert str(err.value) == "Negative ontime encountered."

    def test_frame_distribution(self):
        em = emitter.LooseEmitterSet(xyz=torch.Tensor([[1., 2., 3.], [7., 8., 9.]]), intensity=torch.Tensor([1., 2.]),
                                     t0=torch.Tensor([-0.5, 3.2]), ontime=torch.Tensor([0.4, 2.]),
                                     id=torch.tensor([0, 1]),
                                     sanity_check=True, xy_unit='px', px_size=None)

        """Distribute"""
        xyz, phot, frame_ix, id = em._distribute_framewise()
        # sort by id then by frame_ix
        ix = np.lexsort((id, frame_ix))
        id = id[ix]
        xyz = xyz[ix, :]
        phot = phot[ix]
        frame_ix = frame_ix[ix]

        """Assert"""
        assert (xyz[0] == torch.Tensor([1., 2., 3.])).all()
        assert id[0] == 0
        assert frame_ix[0] == -1
        assert phot[0] == 0.4 * 1

        assert (xyz[1:4] == torch.Tensor([7., 8., 9.])).all()
        assert (id[1:4] == 1).all()
        assert (frame_ix[1:4] == torch.Tensor([3, 4, 5])).all()
        assert test_utils.tens_almeq(phot[1:4], torch.tensor([0.8 * 2, 2, 0.2 * 2]), 1e-6)

    @pytest.fixture()
    def dummy_set(self):
        num_emitters = 10000
        t0_max = 5000
        em = emitter.LooseEmitterSet(torch.rand((num_emitters, 3)), torch.ones(num_emitters) * 10000,
                                     torch.rand(num_emitters) * 3, torch.rand(num_emitters) * t0_max, None,
                                     xy_unit='px', px_size=None)

        return em

    def test_distribution(self):
        loose_em = emitter.LooseEmitterSet(torch.zeros((2, 3)), torch.tensor([1000., 10]), torch.tensor([1., 5]),
                                           torch.tensor([-0.2, 0.9]), xy_unit='px', px_size=None)

        em = loose_em.return_emitterset()
