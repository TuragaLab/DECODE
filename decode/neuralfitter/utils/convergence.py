from abc import ABC, abstractmethod


class ConvergenceCheck(ABC):

    def __call__(self, *args, **kwargs) -> bool:
        return self.check_convergence(*args, **kwargs)

    @abstractmethod
    def check_convergence(self) -> bool:
        """Returns true when convergence seems okay."""
        raise NotImplementedError


class NoCheck(ConvergenceCheck):

    def check_convergence(self) -> bool:
        return True


class GMMHeuristicCheck(ConvergenceCheck):
    def __init__(self, ref_epoch: int, emitter_avg: float, threshold: float = 100.):
        """
        Checks convergence of training by some heuristics.

        Args:
            emitter_avg: Expected number of emitters per frame

        """

        super().__init__()
        self.ref_epoch = ref_epoch
        self.emitter_avg = emitter_avg
        self.threshold = threshold

        self._prev_converged = None  # has already set to converged

    def check_convergence(self, gmm_loss, epoch) -> bool:
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
