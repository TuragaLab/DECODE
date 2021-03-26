from abc import ABC, abstractmethod


class ProgressCheck(ABC):

    def __call__(self, *args, **kwargs) -> bool:
        return self.check_progress(*args, **kwargs)

    @abstractmethod
    def check_progress(self) -> bool:
        """Returns true when convergence seems okay."""
        raise NotImplementedError


class NoCheck(ProgressCheck):

    def check_progress(self) -> bool:
        return True


class GMMHeuristicCheck(ProgressCheck):
    def __init__(self, emitter_avg: float, threshold: float = 100., ref_epoch: int = 1):
        """
        Validates progress of training by some heuristics.

        Args:

            emitter_avg: Expected number of emitters per frame
            threshold: maximum loss per emitter after reference epoch
            ref_epoch: reference epoch

        """

        super().__init__()

        self.emitter_avg = emitter_avg
        self.threshold = threshold
        self.ref_epoch = ref_epoch

        self._prev_converged = None  # has already set to converged

    def check_progress(self, gmm_loss, epoch) -> bool:
        if self._prev_converged is not None:  # do not check if checked before in ref epoch
            return self._prev_converged

        if epoch != self.ref_epoch:
            return True
        else:  # real check happens here
            if gmm_loss / self.emitter_avg >= self.threshold:
                self._prev_converged = False
                return False
            else:
                self._prev_converged = True
                return True
