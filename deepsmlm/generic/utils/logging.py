
def log_metric(tb, cml, value, step, label):
    tb.add_scalar(label, value, step)
    cml.log_metric(label, value, step)

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