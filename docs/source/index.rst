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
We strongly recommend using a machine with a modern CUDA enabled GPU, e.g. an RTX 2080Ti / RTX 3080.
Though, the algorithm will work on non-GPU machines as well (you won't have fun though).
The easiest way to install DECODE, is by using conda.
If you don't have conda (chances are you have it when you have used python) you may download it from https://anaconda.org.
In the following we will make a conda environment and install decode.

Installation in Terminal (macOS, Linux, Anaconda Prompt on Windows)
*******************************************************************

If you are familiar with using conda in your terminal or the Anaconda prompt that's great! Depending on your operating system type:
::
    # Windows, Linux
    conda create -n decode_env -c turagalab -c pytorch -c conda-forge decode cudatoolkit=10.1 jupyterlab

    # macOS (i.e. no CUDA support)
    conda create -n decode_env -c turagalab -c pytorch -c conda-forge decode jupyterlab

    # after previous command (all platforms)
    conda activate decode_env

Installation in Anaconda Navigator (Windows)
**************************************************

You can also use *Anaconda Navigator*.
Create a new environment named *decode_env*, add the channels *turagalab*, *pytorch* and *conda-forge*. 
Add the packages *decode*, *cudatoolkit=10.1* and *jupyterlab*.
Either way, this will create a new environment and install this framework. If everything worked until here you are good
and can skip the next step.


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



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
