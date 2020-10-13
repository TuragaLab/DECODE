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
If you are familiar with using conda in your terminal of choice that's great! Just type:
::
    conda create -n decode_env -c Turagalab -c pytorch -c conda-forge decode python=3.8  # python >=3.6 supported
    conda activate decode_env

This will create a new environment and install this framework. If everything worked until here you are good and can skip the next step.
(optional) You can also create a new environment in the anaconda navigator and install the package there.
Installing the package without creating a new environment (or in a fresh environment) is possible as
::
    conda install -c Turagalab -c pytorch -c conda-forge decode

In python you can now import this package as simple as ``import decode``

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
