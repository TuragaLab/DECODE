"""
DECODE
This software package implements a DeepLearning based framework for high-density fitting in SMLM.

"""

__version__ = '0.9.3.a0'
__author__ = 'Lucas-Raphael Müller, Artur Speiser'
__repo__ = 'https://github.com/TuragaLab/DECODE/master/gateway.yaml'  # main repo
__gateway__ = 'https://raw.githubusercontent.com/TuragaLab/DECODE/master/gateway.yaml'  # gateway

import decode.generic
import decode.evaluation
import decode.neuralfitter
import decode.renderer
import decode.simulation
import decode.plot

from decode.generic.emitter import EmitterSet, RandomEmitterSet, CoordinateOnlyEmitter
