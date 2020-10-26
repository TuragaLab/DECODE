# DECODE

DECODE is a Python and [Pytorch](http://pytorch.org/) based deep learning tool for single molecule localization microscopy (SMLM). It has high accuracy for a large range of imaging modalities and conditions. On the public [SMLM 2016](http://bigwww.epfl.ch/smlm/challenge2016/) software benchmark competition, it [outperformed](http://bigwww.epfl.ch/smlm/challenge2016/leaderboard.html) all other fitters on 12 out of 12 data-sets when comparing both detection accuracy and localization error, often by a substantial margin. DECODE enables live-cell SMLM data with reduced light exposure in just 3 seconds and to image microtubules at ultra-high labeling density.

DECODE works by training a DEep COntext DEpendent (DECODE) neural network to detect and localize emitters at sub-pixel resolution. Notably, DECODE also predict detection and localization uncertainties, which can be used to generate superior super-resolution reconstructions.

## Getting started

The easiest way to try out the algorithm is to have a look at the Google Colab Notebooks we provide for training our algorithm and fitting experimental data. For installation instructions and further information please **refer to our** [**docs**](https://decode.readthedocs.io). 
You can find these here:
- DECODE Training [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/18V1TLLu63CXSWihwoGX7ZQ5wj0Qk7GnD?usp=sharing)
- DECODE Fitting [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1O0mjuOjaOl0wnLZ11Xo92IsWrgqtXL17?usp=sharing)
- [Documentation](https://decode.readthedocs.io)



## Local Installation

We highly recommend using [Anaconda Python](http://anaconda.org) for installing DECODE and its dependencies, including [Pytorch](http://pytorch.org/). While a CUDA-compatible GPU is not essential, it is strongly recommended, as it significantly speeds up the training and analysis steps.

### Installation in Terminal (macOS, Linux, Anaconda Prompt on Windows)
On macOS and Linux please open your terminal, on Windows open Anaconda Prompt. We recommend to set the conda channel_priority to strict. This does two things: Installation is faster, packages are used from the same channel if present. Depending on whether you have a CUDA capable GPU type:

    # (optional, recommended, only do once) weight channel hierarchy more than package version
    conda config --set channel_priority strict

    # CUDA capable GPU
    conda create -n decode_env -c turagalab -c pytorch -c conda-forge decode cudatoolkit=10.1 jupyterlab ipykernel

    # macOS or no CUDA capable GPU
    conda create -n decode_env -c turagalab -c pytorch -c conda-forge decode jupyterlab ipykernel

    # after previous command (all platforms)
    conda activate decode_env

### Installation as package in current environment
Installing the package without creating a new environment (or in a fresh environment) is possible as

    conda install -c turagalab -c pytorch -c conda-forge decode

### Updating
 Please execute the following command in your terminal/Anaconda prompt or do it via the Anaconda Navigator GUI.

    conda update -c turagalab -c pytorch -c conda-forge decode


Please refer to our [docs](https://decode.readthedocs.io) for more information on other options for installation.


## Paper
This is the *official* implementation of the preprint (Link to ArXiV).

Artur Speiser*, Lucas-Raphael Müller*, Ulf Matti, Christopher J. Obara, Wesley R. Legant, Anna Kreshuk, Jakob H. Macke†, Jonas Ries†, and Srinivas C. Turaga†, **Deep learning enables fast and dense single-molecule localization with high accuracy**.

### Data availability
The data referred to in our paper can be accessed at the following locations:
- Fig 3: Can be downloaded from the SMLM 2016 challenge [website](http://bigwww.epfl.ch/smlm/challenge2016/)
- Fig 4: [here](https://oc.embl.de/index.php/s/SFM6Pc8RetX09pJ)
- Fig 5: By request from the authors Wesley R Legant, Lin Shao, Jonathan B Grimm, Timothy A Brown, Daniel E Milkie, Brian B Avants, Luke D Lavis & Eric Betzig, [**High-density three-dimensional localization microscopy across large volumes**](https://www.nature.com/articles/nmeth.3797), _Nature Methods_, *13*, pages 359–365 (2016).

## Contributors
- Artur Speiser ([@aspeiser](https://github.com/ASpeiser), arturspeiser@gmail.com)
- Lucas-Raphael Müller ([@haydnspass](https://github.com/Haydnspass), lucasraphael.mueller@gmail.com)

### Acknowledgements
- Don Olbris ([@olbris](https://github.com/olbris), olbrisd@janelia.hhmi.org) for help with python packaging.

