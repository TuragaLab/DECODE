import math
from abc import abstractmethod
from functools import partial


class GenericScheduler(object):
    """Step an external function when a metric has stopped improving.
    """

    def __init__(self, threshold,
                 mode='min', patience=10, verbose=False, threshold_mode='rel', cooldown=0):

        self.patience = patience
        self.verbose = verbose
        self.cooldown = cooldown
        self.cooldown_counter = 0
        self.mode = mode
        self.threshold = threshold
        self.threshold_mode = threshold_mode
        self.best = None
        self.num_bad_epochs = None
        self.mode_worse = None  # the worse value for the chosen mode
        self.is_better = None
        self.last_epoch = -1
        self._init_is_better(mode=mode, threshold=threshold,
                             threshold_mode=threshold_mode)
        self._reset()

    def _reset(self):
        """Resets num_bad_epochs counter and cooldown counter."""
        self.best = self.mode_worse
        self.cooldown_counter = 0
        self.num_bad_epochs = 0

    def step(self, metrics, epoch=None):
        current = metrics
        if epoch is None:
            epoch = self.last_epoch = self.last_epoch + 1
        self.last_epoch = epoch

        if self.is_better(current, self.best):
            self.best = current
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1

        if self.in_cooldown:
            self.cooldown_counter -= 1
            self.num_bad_epochs = 0  # ignore any bad epochs in cooldown

        if self.num_bad_epochs > self.patience:
            self._do_step(epoch)
            self.cooldown_counter = self.cooldown
            self.num_bad_epochs = 0

    @abstractmethod
    def _do_step(self, epoch):
        pass

    @property
    def in_cooldown(self):
        return self.cooldown_counter > 0

    def _cmp(self, mode, threshold_mode, threshold, a, best):
        if mode == 'min' and threshold_mode == 'rel':
            rel_epsilon = 1. - threshold
            return a < best * rel_epsilon

        elif mode == 'min' and threshold_mode == 'abs':
            return a < best - threshold

        elif mode == 'max' and threshold_mode == 'rel':
            rel_epsilon = threshold + 1.
            return a > best * rel_epsilon

        else:  # mode == 'max' and epsilon_mode == 'abs':
            return a > best + threshold

    def _init_is_better(self, mode, threshold, threshold_mode):
        if mode not in {'min', 'max'}:
            raise ValueError('mode ' + mode + ' is unknown!')
        if threshold_mode not in {'rel', 'abs'}:
            raise ValueError('threshold mode ' + threshold_mode + ' is unknown!')

        if mode == 'min':
            self.mode_worse = math.inf
        else:  # mode == 'max':
            self.mode_worse = -math.inf

        self.is_better = partial(self._cmp, mode, threshold_mode, threshold)

    def state_dict(self):
        return {key: value for key, value in self.__dict__.items() if key not in {'optimizer', 'is_better'}}


class ScheduleSimulation(GenericScheduler):
    def __init__(self, prior, datasets, optimiser, threshold, step_size, max_emitter,
                 mode='min', patience=10, threshold_mode='rel', cooldown=0, verbose=True, disabled=False):
        """

        :param prior:
        :param datasets:
        :param optimiser:
        :param threshold:
        :param step_size:
        :param max_emitter:
        :param mode:
        :param patience:
        :param threshold_mode:
        :param cooldown:
        :param verbose:
        :param disabled:
        """
        super().__init__(threshold,
                         mode=mode,
                         patience=patience,
                         verbose=verbose,
                         threshold_mode=threshold_mode,
                         cooldown=cooldown)

        self.prior = prior
        self.datasets = datasets
        self.optimiser = optimiser
        self.init_lr = None
        self.max_emitter_av = max_emitter
        self.step_size = step_size
        self.verbose = verbose
        self.disabled = disabled

        self.init_lr = optimiser.param_groups[0]['lr']

    def _do_step(self, epoch):

        if self.disabled:
            return

        if self.prior.emitter_av < self.max_emitter_av:
            self.prior.emitter_av *= self.step_size
            print("Increased complexitiy. New emitter average: {}".format(self.prior.emitter_av))
            for dataset in self.datasets:
                dataset.drop_data_set(self.verbose)

            """If complexity increased, reset the learning rate"""
            for o in range(self.optimiser.param_groups.__len__()):
                self.optimiser.param_groups[o]['lr'] = self.init_lr
            print("Reset learning rate to original value.")

        else:
            print("Maximum complexity reached.")


if __name__ == '__main__':
    x = 1

    scheduler = ScheduleSimulation(None, 0.01, 1, 10)
    scheduler.step(x)
    for i in range(1000):
        print(i)
        scheduler.step(x)
