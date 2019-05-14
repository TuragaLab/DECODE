
def log_metric(tb, cml, value, step, label):
    tb.add_scalar(label, value, step)
    cml.log_metric(label, value, step)