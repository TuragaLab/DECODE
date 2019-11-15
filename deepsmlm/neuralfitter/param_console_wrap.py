import os
import sys
import getopt

"""
Parsing parameters from command line. 
Valid options are p for specifying the parameter file and n for not logging the stuff.
"""
param_file = None
unixOptions = "p:n:g"
gnuOptions = ["params=", "no_log", "gpus="]

# read commandline arguments, first
argumentList = sys.argv[1:]

try:
    arguments, values = getopt.getopt(argumentList, unixOptions, gnuOptions)
except getopt.error as err:
    # output error, and return with an error code
    print(str(err))
    sys.exit(2)

for arg, argV in arguments:
    if arg in ("-n", "--no_log"):
        WRITE_TO_LOG = False
        print("Not logging the current run.")

    elif arg in ("-p", "--params"):
        print("Parameter file is in: {}".format(argV))
        param_file = argV

    elif arg in ("-g", "--gpus"):
        cuda_str = [(str(gpu_ix) + ',') for gpu_ix in argV]
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_str