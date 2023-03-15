============
Installation
============

For regular use, we advise you to install and use the framework on your
local machine. We strongly recommend using a machine with a modern GPU, e.g. an
RTX 2080, in particular for training. To make use of your GPU it requires a CUDA
capability of 3.7 or higher (see here to check if your GPU is valid:
https://en.wikipedia.org/wiki/CUDA#GPUs\_supported). However, the algorithm will
work on non-GPU machines as well (you won't have fun though). The easiest way to
install DECODE, is by using conda. If you don't have conda (chances are you have
it when you have used python) you may download it from https://anaconda.org. In
the following we will make a conda environment and install decode.

System Requirements
-------------------

-  GPU: CUDA with RAM >= 4GB and compute capability >= 3.7 (both highly recommended)
-  RAM: >= 8 GB
-  CPU: Multi-Core recommended
-  OS: Linux (GPU accelerated), Windows (GPU accelerated), macOS (CPU only)
-  Software: conda, anaconda

Installation in Terminal (macOS, Linux, Anaconda Prompt on Windows)
-------------------------------------------------------------------

Note that in the following we install *jupyterlab* (and *ipykernel*) in addition
to DECODE. This is not strictly needed to run the code locally, but you need some
kind of jupyter lab instance with the decode environment enabled in it to run the
examples as described in the Tutorial.

On macOS and Linux please open your terminal, on Windows open *Anaconda Prompt*.
We recommend to set the conda channel_priority to strict. This does two things:
Installation is faster, packages are used from the same channel if present.

Depending on whether you have a CUDA capable GPU type:

.. code:: bash

    # (optional, recommended, only do once) weight channel hierarchy more than package version
    conda config --set channel_priority strict

    # CUDA capable GPU
    conda create -n decode_env -c turagalab -c pytorch -c conda-forge decode=0.10.2 cudatoolkit=11.3 jupyterlab ipykernel

    # macOS or no CUDA capable GPU
    conda create -n decode_env -c turagalab -c pytorch -c conda-forge decode=0.10.2 jupyterlab ipykernel

    # after previous command (all platforms)
    conda activate decode_env

Please now get the DECODE Jupyter Notebooks.

.. _notebook_install:

DECODE Jupyter Notebooks
""""""""""""""""""""""""

Before you start using DECODE locally, you should make sure to check out our Jupyter notebooks
to familiarise yourself with DECODE.
You can get the notebooks by specifying the directory where you want the notebooks to be saved following this
command in your Terminal/Anaconda Prompt:

.. code:: bash

    conda activate decode_env

    # get the example notebooks
    python -m decode.utils.notebooks [e.g. /Users/RainerZufall/Downloads]

In case of trouble please refer to the `Instructions for JupyterLab <https://jupyterlab.readthedocs.io/en/stable/getting_started/installation.html>`__.
You are now all set to start the `DECODE tutorial. <tutorial.html>`__


Updating
--------

Please execute the following command in your terminal/Anaconda prompt.

.. code:: bash

    conda update -n decode_env -c turagalab -c pytorch -c conda-forge decode

The Jupyter notebooks are coupled to the version of DECODE you have installed. A version mismatch
might lead to non-functional notebooks. Please get a fresh copy of the notebooks by simply
running following the :ref:`instructions to get the DECODE notebooks.<notebook_install>`


Import
------

In python you can import this package as simple as ``import decode``. You may
continue with our `tutorial <./tutorial.html>`__.
