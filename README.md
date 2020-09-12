# DECODE
**Deep learning enables fast and dense single-molecule localization with high accuracy**

This is the *official* implementation of the preprint (Link to ArXiV). 
The easiest way to try out the algorithm is to have a look at the Google Colab Notebooks we provide for training our algorithm and fitting experimental data. You can find these notebooks here:
- DECODE Training [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/18V1TLLu63CXSWihwoGX7ZQ5wj0Qk7GnD?usp=sharing)
- DECODE Fitting [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1O0mjuOjaOl0wnLZ11Xo92IsWrgqtXL17?usp=sharing)

You can find the documentation by clicking on the 'Wiki' button here in github, or under this link: [LINK to Readthedocs].


## Local Installation for Users (local machine)
For regular use, of course we recommend to install and use the framework on your local machine / workstation. We strongly recommend using a machine with a modern CUDA enabled GPU, e.g. an RTX 2080Ti.
Though, the algorithm will work on none-GPU machines as well.
The easiest way to install DECODE, is by using conda. If you don't have conda (chances are you have it when you have used python)
you may download it from https://anaconda.org.

In the following we will make a conda environment and install decode. If you are familiar with using
```conda``` in your terminal of choice that's great! Just type:

        conda create -n decode_env -c haydnspass -c pytorch -c conda-forge decode python=3.8  # python >=3.6 supported
        conda activate decode_env

This will create a new environment and install this framework.
If everything worked until here you are good and can skip the next step.

- **(optional)** You can also create a new environment in the anaconda
navigator and install the package there. Installing the package without creating a new environment
(or in a fresh environment) is possible as

        conda install -c haydnspass -c pytorch -c conda-forge decode

In python you can now import this package as simple as ```import decode```

## Setup for Developers
[![Build Status](https://travis-ci.com/Haydnspass/DeepSMLM.svg?token=qb4PpCab8Gb7CDLAuNTY&branch=master)](https://travis-ci.com/Haydnspass/DeepSMLM)

0. Clone the repository
1. Install conda environment from file and activate it. Use the respective environment depending on whether you have a CUDA GPU or not.
The cubic spline psf is pre-compiled in a different repo / as a separate package and will be installed automatically.

        # CUDA
        conda env create -f environment_cuda102_py38_pt15.yml
        conda activate decode_cuda_py38_pt15   

        # CPU / macOS
        conda env create -f environment_cpu_py38_pt15.yml
        conda activate deepsmlm_cpu

3. Test whether everything works as expected

            # assuming you are in the repo folder
            pytest decode/test

            # or if you fancy some nice figures, depending on your IDE
            # you might need to close popping up matplot figures
            pytest decode/test --plot  

4. The package can be used in python as

    ```import decode```

### Note
Simulation heavily relies on the Cubic Spline PSF Implementation (Li, Y. et al. Real-time 3D single-molecule localization using experimental point spread functions. Nature Methods 15, 367–369 (2018)).
It has been factored out in a seperate Repo to make life easier (see https://github.com/Haydnspass/SplinePSF). It'll be auto-installed.

### Building the Docs
Create docs environment from file `environment_docs.yaml`