import matplotlib.pyplot as plt

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

    def forward(self, metrics_set, step):
        self._log_metric(metrics_set.prec.avg, step, "eval/precision")
        self._log_metric(metrics_set.rec.avg, step, "eval/recall")
        self._log_metric(metrics_set.jac.avg, step, "eval/jac")

        self._log_metric(metrics_set.rmse_vol.avg, step, "eval/rmse_vol")
        self._log_metric(metrics_set.rmse_lat.avg, step, "eval/rmse_lat")
        self._log_metric(metrics_set.rmse_axial.avg, step, "eval/rmse_axial")

        self._log_metric(metrics_set.mad_vol.avg, step, "eval/mad_vol")
        self._log_metric(metrics_set.mad_lat.avg, step, "eval/mad_lat")
        self._log_metric(metrics_set.mad_axial.avg, step, "eval/mad_axial")

        _ = metrics_set.dx.hist()
        plt.gca().set_xlabel(r'$dx$')
        self._log_figure(plt.gcf(), step, "eval/dx")
        # self.tb.add_histogram('eval.dx', metrics_set.dx.vals.numpy(), step)

        _ = metrics_set.dy.hist()
        plt.gca().set_xlabel(r'$dy$')
        self._log_figure(plt.gcf(), step, "eval/dy")
        # self.tb.add_histogram('eval.dx', metrics_set.dy.vals.numpy(), step)

        _ = metrics_set.dz.hist()
        plt.gca().set_xlabel(r'$dz$')
        self._log_figure(plt.gcf(), step, "eval/dz")
        # self.tb.add_histogram('eval.dz', metrics_set.dz.vals.numpy(), step)

        _ = metrics_set.dxw.hist()
        plt.gca().set_xlabel(r'$dx \cdot \sqrt{N}$')
        self._log_figure(plt.gcf(), step, "eval/dxw")
        # self.tb.add_histogram('eval.dxw', metrics_set.dxw.vals.numpy(), step)

        _ = metrics_set.dyw.hist()
        plt.gca().set_xlabel(r'$dy \cdot \sqrt{N}$')
        self._log_figure(plt.gcf(), step, "eval/dyw")
        # self.tb.add_histogram('eval.dyw', metrics_set.dyw.vals.numpy(), step)

        _ = metrics_set.dzw.hist()
        plt.gca().set_xlabel(r'$dz \cdot \sqrt{N}$')
        self._log_figure(plt.gcf(), step, "eval/dzw")
        # self.tb.add_histogram('eval.dzw', metrics_set.dzw.vals.numpy(), step)

