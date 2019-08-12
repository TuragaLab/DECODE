import matplotlib.pyplot as plt
import torch
import random

import deepsmlm.generic.plotting.frame_coord as emplot
from deepsmlm.evaluation.evaluation import MetricMeter


class LogTestEpoch:
    """
    Log relevant metrics to both tensorboard and comet.
    """
    def __init__(self, tb_instance, cm_instance):
        self.tb = tb_instance
        self.cml = cm_instance

    def _log_metric(self, value, step, label):
        self.tb.add_scalar(label, value, step)
        self.cml.log_metric(label, value, step)

    def _log_figure(self, fig, step, label):
        self.tb.add_figure(label, fig, step)
        self.cml.log_figure(label, fig)

    def forward(self, metrics_set, input_frames, output_frames, target_frames, em_out, em_tar, step):
        self._log_metric(metrics_set.prec.avg, step, "eval/precision")
        self._log_metric(metrics_set.rec.avg, step, "eval/recall")
        self._log_metric(metrics_set.jac.avg, step, "eval/jac")

        self._log_metric(metrics_set.rmse_vol.avg, step, "eval/rmse_vol")
        self._log_metric(metrics_set.rmse_lat.avg, step, "eval/rmse_lat")
        self._log_metric(metrics_set.rmse_axial.avg, step, "eval/rmse_axial")

        self._log_metric(metrics_set.mad_vol.avg, step, "eval/mad_vol")
        self._log_metric(metrics_set.mad_lat.avg, step, "eval/mad_lat")
        self._log_metric(metrics_set.mad_axial.avg, step, "eval/mad_axial")

        self._log_metric(metrics_set.delta_num.avg, step, "eval/del_num_em")

        """Plot input / output. Random sample."""
        ix = random.randint(0, output_frames.shape[0] - 1)
        em_tar_ = em_tar[ix]
        em_out_ = em_out[ix]

        _ = plt.figure(figsize=(12, 6))
        plt.subplot(131)
        emplot.PlotFrameCoord(frame=input_frames[ix, 0],
                              pos_tar=em_tar_.xyz,
                              phot_tar=em_tar_.phot).plot()
        plt.subplot(132)
        emplot.PlotFrameCoord(frame=input_frames[ix, 1],
                              pos_tar=em_tar_.xyz,
                              phot_tar=em_tar_.phot).plot()
        plt.subplot(133)
        emplot.PlotFrameCoord(frame=input_frames[ix, 2],
                              pos_tar=em_tar_.xyz,
                              phot_tar=em_tar_.phot).plot()

        self._log_figure(plt.gcf(), step, "io/frames_in")

        """Plot Target"""
        # keep index
        _ = plt.figure(figsize=(12, 12))
        plt.subplot(231)
        emplot.PlotFrameCoord(frame=input_frames[ix, 1],
                              pos_tar=em_tar_.xyz,
                              phot_tar=em_tar_.phot).plot();
        plt.legend()
        plt.subplot(232)
        emplot.PlotFrame(frame=target_frames[ix, 0]).plot()
        plt.title('p channel')
        plt.subplot(233)
        emplot.PlotFrame(frame=target_frames[ix, 1]).plot()
        plt.title('phot channel')
        plt.subplot(234)
        emplot.PlotFrame(frame=target_frames[ix, 2]).plot()
        plt.title('dx channel')
        plt.subplot(235)
        emplot.PlotFrame(frame=target_frames[ix, 3]).plot()
        plt.title('dy channel')
        plt.subplot(236)
        emplot.PlotFrame(frame=target_frames[ix, 4]).plot()
        plt.title('z channel')

        self._log_figure(plt.gcf(), step, "io/target")

        _ = plt.figure(figsize=(12, 12))
        plt.subplot(231)
        emplot.PlotFrameCoord(frame=input_frames[ix, 1],
                              pos_tar=em_tar_.xyz,
                              phot_tar=em_tar_.phot).plot(); plt.legend()
        plt.title('Input')

        plt.subplot(232)
        emplot.PlotFrameCoord(frame=output_frames[ix, 0], clim=(0., 1.)).plot()
        plt.title('p channel')

        plt.subplot(233)
        emplot.PlotFrame(frame=output_frames[ix, 1]).plot()
        plt.title('phot channel')

        plt.subplot(234)
        emplot.PlotFrame(frame=output_frames[ix, 2]).plot()
        plt.title('dx channel')

        plt.subplot(235)
        emplot.PlotFrame(frame=output_frames[ix, 3]).plot()
        plt.title('dy channel')

        plt.subplot(236)
        emplot.PlotFrame(frame=output_frames[ix, 4]).plot()
        plt.title('z channel')

        self._log_figure(plt.gcf(), step, "io/tarframe_output")

        """Raw output distributions where we have signal above a threshold"""
        th = 0.1
        is_above_th = output_frames[:, 0] > th

        phot_dist = MetricMeter()
        dx_dist, dy_dist, z_dist = MetricMeter(), MetricMeter(), MetricMeter()

        phot_dist.vals = output_frames[:, 1][is_above_th]
        dx_dist.vals = output_frames[:, 2][is_above_th]
        dy_dist.vals = output_frames[:, 3][is_above_th]
        z_dist.vals = output_frames[:, 4][is_above_th]

        _ = phot_dist.hist(fit=None, range=(0., 1.))
        plt.gca().set_xlabel(r'$phot$')
        self._log_figure(plt.gcf(), step, "io/phot")

        _ = dx_dist.hist(fit=None)
        plt.gca().set_xlabel(r'$dx$')
        self._log_figure(plt.gcf(), step, "io/dx")

        _ = dy_dist.hist(fit=None)
        plt.gca().set_xlabel(r'$dy$')
        self._log_figure(plt.gcf(), step, "io/dy")

        _ = z_dist.hist(fit=None)
        plt.gca().set_xlabel(r'$z$')
        self._log_figure(plt.gcf(), step, "io/z")

        """Dx/y/z histograms"""
        _ = metrics_set.dx.hist()
        plt.gca().set_xlabel(r'$\Delta x$')
        self._log_figure(plt.gcf(), step, "eval/dx")
        # self.tb.add_histogram('eval.dx', metrics_set.dx.vals.numpy(), step)

        _ = metrics_set.dy.hist()
        plt.gca().set_xlabel(r'$\Delta y$')
        self._log_figure(plt.gcf(), step, "eval/dy")
        # self.tb.add_histogram('eval.dx', metrics_set.dy.vals.numpy(), step)

        _ = metrics_set.dz.hist()
        plt.gca().set_xlabel(r'$\Delta z$')
        self._log_figure(plt.gcf(), step, "eval/dz")
        # self.tb.add_histogram('eval.dz', metrics_set.dz.vals.numpy(), step)

        _ = metrics_set.dxw.hist()
        plt.gca().set_xlabel(r'$\Delta x \cdot \sqrt{N}$')
        self._log_figure(plt.gcf(), step, "eval/dxw")
        # self.tb.add_histogram('eval.dxw', metrics_set.dxw.vals.numpy(), step)

        _ = metrics_set.dyw.hist()
        plt.gca().set_xlabel(r'$\Delta y \cdot \sqrt{N}$')
        self._log_figure(plt.gcf(), step, "eval/dyw")
        # self.tb.add_histogram('eval.dyw', metrics_set.dyw.vals.numpy(), step)

        _ = metrics_set.dzw.hist()
        plt.gca().set_xlabel(r'$\Delta z \cdot \sqrt{N}$')
        self._log_figure(plt.gcf(), step, "eval/dzw")
        # self.tb.add_histogram('eval.dzw', metrics_set.dzw.vals.numpy(), step)

