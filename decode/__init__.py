"""
DECODE
This software package implements a DeepLearning based framework for high-density fitting in SMLM.

"""

__version__ = '0.9.a'
__author__ = 'Lucas-Raphael Müller, Artur Speiser'

import decode.generic
import decode.evaluation
import decode.neuralfitter
import decode.renderer
import decode.simulation
import decode.plot

from decode.generic.emitter import EmitterSet, RandomEmitterSet, CoordinateOnlyEmitter
