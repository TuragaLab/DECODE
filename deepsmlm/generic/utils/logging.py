import matplotlib.pyplot as plt
import torch
import random

import deepsmlm.generic.plotting.frame_coord as emplot
from deepsmlm.evaluation.utils import MetricMeter


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

    def _log_figure(self, fig, step, label, show=False):
        if show:
            plt.show(fig=fig)
            return

        self.tb.add_figure(label, fig, step)
        self.cml.log_figure(label, fig)

    def forward(self, metrics_set, input_frames, output_frames, target_frames, em_out, em_tar, step,
                weight_frames=None, show=False):
        self._log_metric(metrics_set.prec.avg, step, "eval/precision")
        self._log_metric(metrics_set.rec.avg, step, "eval/recall")
        self._log_metric(metrics_set.jac.avg, step, "eval/jac")
        self._log_metric(metrics_set.f1.avg, step, "eval/f1")

        self._log_metric(metrics_set.rmse_vol.avg, step, "eval/rmse_vol")
        self._log_metric(metrics_set.rmse_lat.avg, step, "eval/rmse_lat")
        self._log_metric(metrics_set.rmse_axial.avg, step, "eval/rmse_axial")

        self._log_metric(metrics_set.mad_vol.avg, step, "eval/mad_vol")
        self._log_metric(metrics_set.mad_lat.avg, step, "eval/mad_lat")
        self._log_metric(metrics_set.mad_axial.avg, step, "eval/mad_axial")

        self._log_metric(metrics_set.effcy_lat.avg, step, "eval/effcy_lat")
        self._log_metric(metrics_set.effcy_ax.avg, step, "eval/effcy_ax")
        self._log_metric(metrics_set.effcy_vol.avg, step, "eval/effcy_vol")

        self._log_metric(metrics_set.delta_num.avg, step, "eval/del_num_em")

        """Compute (true) number of emitters per frame"""
        nem_av = len(em_tar) / (em_tar.frame_ix.max() - em_tar.frame_ix.min() + 1)

        # std
        em_per_frame = em_tar.split_in_frames()
        em_per_frame = [len(e) for e in em_per_frame]
        nem_std = torch.tensor(em_per_frame).float().std().item()

        self._log_metric(nem_av, step, "data/num_em_avg")
        self._log_metric(nem_std, step, "data/num_em_std")

        """Plot input / output. Random sample."""
        ix = random.randint(0, output_frames.shape[0] - 1)
        em_tar_ = em_tar.get_subset_frame(ix, ix)
        em_out_ = em_out.get_subset_frame(ix, ix)

        _ = plt.figure(figsize=(12, 6))
        if input_frames.size(1) == 3:
            plt.subplot(131)
            emplot.PlotFrameCoord(frame=input_frames[ix, 0],
                                  pos_tar=em_tar_.xyz,
                                  phot_tar=em_tar_.phot).plot()
            # plt.colorbar(fraction=0.046, pad=0.04)

            plt.subplot(132)
            emplot.PlotFrameCoord(frame=input_frames[ix, 1],
                                  pos_tar=em_tar_.xyz,
                                  phot_tar=em_tar_.phot).plot()
            # plt.colorbar(fraction=0.046, pad=0.04)

            plt.subplot(133)
            emplot.PlotFrameCoord(frame=input_frames[ix, 2],
                                  pos_tar=em_tar_.xyz,
                                  phot_tar=em_tar_.phot).plot()
            # plt.colorbar(fraction=0.046, pad=0.04)
        else:
            emplot.PlotFrameCoord(frame=input_frames[ix, 0],
                                  pos_tar=em_tar_.xyz,
                                  phot_tar=em_tar_.phot).plot()
            # plt.colorbar(fraction=0.046, pad=0.04)

        self._log_figure(plt.gcf(), step, "io/frames_in", show)

        """Plot Target"""
        if input_frames.size(1) == 3:
            tar_ch = 1
        elif input_frames.size(1) == 1:
            tar_ch = 0

        _ = plt.figure(figsize=(12, 12))

        plt.subplot(231)
        emplot.PlotFrame(frame=target_frames[ix, 0]).plot()
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.title('p channel')
        plt.subplot(232)
        emplot.PlotFrame(frame=target_frames[ix, 1]).plot()
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.title('phot channel')
        plt.subplot(233)
        emplot.PlotFrame(frame=target_frames[ix, 2]).plot()
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.title('dx channel')
        plt.subplot(234)
        emplot.PlotFrame(frame=target_frames[ix, 3]).plot()
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.title('dy channel')
        plt.subplot(235)
        emplot.PlotFrame(frame=target_frames[ix, 4]).plot()
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.title('z channel')
        if target_frames.size(1) >= 6:
            plt.subplot(236)
            emplot.PlotFrame(frame=target_frames[ix, 5]).plot()
            plt.colorbar(fraction=0.046, pad=0.04)
            plt.title('bg channel')

        self._log_figure(plt.gcf(), step, "io/target", show)

        _ = plt.figure(figsize=(12, 12))
        plt.subplot(231)
        emplot.PlotFrameCoord(frame=output_frames[ix, 0], clim=(0., 1.)).plot()
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.title('p channel')

        plt.subplot(232)
        emplot.PlotFrame(frame=output_frames[ix, 1]).plot()
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.title('phot channel')

        plt.subplot(233)
        emplot.PlotFrame(frame=output_frames[ix, 2]).plot()
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.title('dx channel')

        plt.subplot(234)
        emplot.PlotFrame(frame=output_frames[ix, 3]).plot()
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.title('dy channel')

        plt.subplot(235)
        emplot.PlotFrame(frame=output_frames[ix, 4]).plot()
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.title('z channel')

        if target_frames.size(1) >= 6:
            plt.subplot(236)
            emplot.PlotFrame(frame=output_frames[ix, 5]).plot()
            plt.colorbar(fraction=0.046, pad=0.04)
            plt.title('bg channel')

        self._log_figure(plt.gcf(), step, "io/tarframe_output", show)

        if weight_frames is not None:
            _ = plt.figure(figsize=(12, 12))
            plt.subplot(231)
            emplot.PlotFrameCoord(frame=weight_frames[ix, 0], clim=(0., 1.)).plot()
            plt.colorbar(fraction=0.046, pad=0.04)
            plt.title('p channel')

            plt.subplot(232)
            emplot.PlotFrame(frame=weight_frames[ix, 1]).plot()
            plt.colorbar(fraction=0.046, pad=0.04)
            plt.title('phot channel')

            plt.subplot(233)
            emplot.PlotFrame(frame=weight_frames[ix, 2]).plot()
            plt.colorbar(fraction=0.046, pad=0.04)
            plt.title('dx channel')

            plt.subplot(234)
            emplot.PlotFrame(frame=weight_frames[ix, 3]).plot()
            plt.colorbar(fraction=0.046, pad=0.04)
            plt.title('dy channel')

            plt.subplot(235)
            emplot.PlotFrame(frame=weight_frames[ix, 4]).plot()
            plt.colorbar(fraction=0.046, pad=0.04)
            plt.title('z channel')

            plt.subplot(236)
            emplot.PlotFrame(frame=weight_frames[ix, 5]).plot()
            plt.colorbar(fraction=0.046, pad=0.04)
            plt.title('bg channel')

            self._log_figure(plt.gcf(), step, "io/weight_map", show)

        """Raw distribution of the input frames"""
        in_dist = MetricMeter()
        in_dist.vals = input_frames
        _ = in_dist.hist(fit=None)
        plt.gca().set_xlabel(r'raw_in')
        self._log_figure(plt.gcf(), step, "io/raw_in", show)

        """Raw output distributions where we have signal above a threshold"""
        th = 0.05
        is_above_th = output_frames[:, 0] > th

        phot_dist = MetricMeter()
        dx_dist, dy_dist, z_dist = MetricMeter(), MetricMeter(), MetricMeter()
        bg_dist_out, bg_dist_tar = MetricMeter(), MetricMeter()
        bg_dist_out_rsample, bg_dist_tar_rsample = MetricMeter(), MetricMeter()

        phot_dist.vals = output_frames[:, 1][is_above_th]
        dx_dist.vals = output_frames[:, 2][is_above_th]
        dy_dist.vals = output_frames[:, 3][is_above_th]
        z_dist.vals = output_frames[:, 4][is_above_th]
        bg_dist_out.vals = output_frames[:, 5].reshape(-1)
        bg_dist_tar.vals = target_frames[:, 5].reshape(-1)
        bg_dist_out_rsample.vals = output_frames[ix, 5].reshape(-1)
        bg_dist_tar_rsample.vals = target_frames[ix, 5].reshape(-1)

        _ = phot_dist.hist(fit=None, range_hist=(0., 1.))
        plt.gca().set_xlabel(r'$phot$')
        self._log_figure(plt.gcf(), step, "io/phot", show)

        _ = dx_dist.hist(fit=None)
        plt.gca().set_xlabel(r'$dx$')
        self._log_figure(plt.gcf(), step, "io/dx", show)

        _ = dy_dist.hist(fit=None)
        plt.gca().set_xlabel(r'$dy$')
        self._log_figure(plt.gcf(), step, "io/dy", show)

        _ = z_dist.hist(fit=None)
        plt.gca().set_xlabel(r'$z$')
        self._log_figure(plt.gcf(), step, "io/z", show)

        _ = bg_dist_out.hist(fit=None)
        plt.gca().set_xlabel(r'$bg_{out}$')
        self._log_figure(plt.gcf(), step, "io/bg_out", show)

        _ = bg_dist_tar.hist(fit=None)
        plt.gca().set_xlabel(r'$bg_{tar}$')
        self._log_figure(plt.gcf(), step, "io/bg_tar", show)

        _ = bg_dist_out_rsample.hist(fit=None)
        plt.gca().set_xlabel(r'$bg_{out,rsample}$')
        self._log_figure(plt.gcf(), step, "io/bg_out_rsample", show)

        _ = bg_dist_tar_rsample.hist(fit=None)
        plt.gca().set_xlabel(r'$bg_{tar,rsample}$')
        self._log_figure(plt.gcf(), step, "io/bg_tar_rsample", show)

        """Dx/y/z histograms"""
        _ = metrics_set.dx.hist()
        plt.gca().set_xlabel(r'$\Delta x$')
        self._log_figure(plt.gcf(), step, "eval/dx", show)
        # self.tb.add_histogram('eval.dx', metrics_set.dx.vals.numpy(), step)

        _ = metrics_set.dy.hist()
        plt.gca().set_xlabel(r'$\Delta y$')
        self._log_figure(plt.gcf(), step, "eval/dy", show)
        # self.tb.add_histogram('eval.dx', metrics_set.dy.vals.numpy(), step)

        _ = metrics_set.dz.hist()
        plt.gca().set_xlabel(r'$\Delta z$')
        self._log_figure(plt.gcf(), step, "eval/dz", show)
        # self.tb.add_histogram('eval.dz', metrics_set.dz.vals.numpy(), step)

        _ = metrics_set.dxw.hist()
        plt.gca().set_xlabel(r'$\Delta x \cdot \sqrt{N}$')
        self._log_figure(plt.gcf(), step, "eval/dxw", show)
        # self.tb.add_histogram('eval.dxw', metrics_set.dxw.vals.numpy(), step)

        _ = metrics_set.dyw.hist()
        plt.gca().set_xlabel(r'$\Delta y \cdot \sqrt{N}$')
        self._log_figure(plt.gcf(), step, "eval/dyw", show)
        # self.tb.add_histogram('eval.dyw', metrics_set.dyw.vals.numpy(), step)

        _ = metrics_set.dzw.hist()
        plt.gca().set_xlabel(r'$\Delta z \cdot \sqrt{N}$')
        self._log_figure(plt.gcf(), step, "eval/dzw", show)
        # self.tb.add_histogram('eval.dzw', metrics_set.dzw.vals.numpy(), step)

