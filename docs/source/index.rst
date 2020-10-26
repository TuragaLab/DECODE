DECODE Documentation
====================================
This is the documentation of the DECODE Deep Learning for Superresolution Localization Microscopy.

Usage and Installation
======================
To try out the package we recommend the Google Colab notebooks which comprise training a model and fitting experimental data.

Google Colab Notebooks
######################
* `Training <https://colab.research.google.com/drive/18V1TLLu63CXSWihwoGX7ZQ5wj0Qk7GnD?usp=sharing>`_
* `Fitting <https://colab.research.google.com/drive/1O0mjuOjaOl0wnLZ11Xo92IsWrgqtXL17?usp=sharing>`_

Local Installation
##################

For regular use, of course we recommend to install and use the framework on your local machine.
We strongly recommend using a machine with a modern GPU, e.g. an RTX 3080, in particular for training.
To make use of your GPU it requires a CUDA capability of 3.7 or higher (see here to check if your GPU is valid: https://en.wikipedia.org/wiki/CUDA#GPUs_supported).
However, the algorithm will work on non-GPU machines as well (you won't have fun though).
The easiest way to install DECODE, is by using conda.
If you don't have conda (chances are you have it when you have used python) you may download it from https://anaconda.org.
In the following we will make a conda environment and install decode.

System Requirements (TLDR)
**************************
* GPU: CUDA with compute capability >= 3.7 (highly recommended)
* OS: Linux (GPU accelerated), Windows (GPU accelerated), macOS (CPU only)
* RAM: >= 8 GB
* CPU: Multi-Core recommended
* Software: conda, anaconda

Installation in Terminal (macOS, Linux, Anaconda Prompt on Windows)
*******************************************************************

Note that in the following we install *jupyterlab* (and *ipykernel*) in addition to DECODE. This is not strictly needed to run the code locally, but you need
some kind of jupyter lab instance with the decode environment enabled in it to run the examples as described in the Tutorial.

On macOS and Linux please open your terminal, on Windows open *Anaconda Prompt*.
We recommend to set the conda channel_priority to strict. This does two things: Installation is faster, packages are used from the same channel if present.

Depending on whether you have a CUDA capable GPU type:
::
    # (optional, recommended, only do once) weight channel hierarchy more than package version
    conda config --set channel_priority strict

    # CUDA capable GPU
    conda create -n decode_env -c turagalab -c pytorch -c conda-forge decode cudatoolkit=10.1 jupyterlab ipykernel

    # macOS or no CUDA capable GPU
    conda create -n decode_env -c turagalab -c pytorch -c conda-forge decode jupyterlab ipykernel

    # after previous command (all platforms)
    conda activate decode_env


Installation as package in current environment
***********************************************
Installing the package without creating a new environment (or in a fresh environment) is possible as
::
    conda install -c turagalab -c pytorch -c conda-forge decode

Updating
***********
Please execute the following command in your terminal/Anaconda prompt or do it via the Anaconda Navigator GUI.
::
    conda update -c turagalab -c pytorch -c conda-forge decode

Import
******* 

In python you can import this package as simple as ``import decode``.
You may continue with our tutorial.

.. toctree::
   :maxdepth: 1
   :caption: Contents


.. toctree::
   :maxdepth: 1
   :caption: DECODE API

   decode

.. toctree::
   :maxdepth: 0
   :caption: Tutorial

   tutorial

.. toctree::
   :maxdepth: 0
   :caption: FAQ

   faq
