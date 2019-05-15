
def log_metric(tb, cml, value, step, label):
    tb.add_scalar(label, value, step)
    cml.log_metric(label, value, step)

class LogTraining:
    def __init__(self):
        pass

    def log_per_train_batch(self):
        pass