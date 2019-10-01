import os, sys
import torch
import pytest

import deepsmlm.generic.emitter as em
import deepsmlm.simulation.simulator as sim
import deepsmlm.generic.inout.load_calibration as load_calib

deepsmlm_root = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)),
                 os.pardir, os.pardir)) + '/'


class TestSimulator:

    @pytest.fixture(scope='class')
    def dummy_em(self):
        return em.EmitterSet(torch.tensor([[0., 0., 0]]),
                             torch.tensor([1000.]),
                             torch.tensor([0.]))

    @pytest.fixture(scope='class')
    def dummy_sim(self):
        psf_extent = ((-0.5, 63.5), (-0.5, 63.5), (-750, 750))
        img_shape = (64, 64)
        csp_calib = deepsmlm_root + \
                    'data/Calibration/2019-06-13_Calibration/sequence-as-stack-Beads-AS-Exp_3dcal.mat'

        sp = load_calib.SMAPSplineCoefficient(csp_calib)
        psf = sp.init_spline(psf_extent[0], psf_extent[1], img_shape=img_shape)
        return sim.Simulation(em=None, extent=None, psf=psf, background=None, frame_range=(-1, 1), poolsize=0)

    def test_forward(self, dummy_em, dummy_sim):
        x = dummy_sim.forward(dummy_em)
        print("Done")
