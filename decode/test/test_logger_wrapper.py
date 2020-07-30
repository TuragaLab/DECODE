import warnings

from decode.neuralfitter.utils import logger


def test_noop_logger():
    logger.NoLog()
    warnings.warn("No proper test implemented.")

