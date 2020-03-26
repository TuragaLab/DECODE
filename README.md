# DEEPsmlm
[![Build Status](https://travis-ci.com/Haydnspass/DeepSMLM.svg?token=qb4PpCab8Gb7CDLAuNTY&branch=master)](https://travis-ci.com/Haydnspass/DeepSMLM)
[![Build Status](https://travis-ci.com/Haydnspass/DeepSMLM.svg?token=qb4PpCab8Gb7CDLAuNTY&branch=dev_decode_repr)](https://travis-ci.com/Haydnspass/DeepSMLM)

## Installation for Users
The easiest way to get it, is by using conda. If you don't have conda (chances are you have it when you have used python)
you may download it from https://anaconda.org.

In the following we will make a conda environment and install deepsmlm. If you are familiar with using
```conda``` in your terminal of choice that's great! Just type:

        conda create -n deepsmlm_env -c haydnspass -c conda-forge deepsmlm
        conda activate deepsmlm_env
        
This will create a new environment and install this framework. 
If everything worked until here you are good and can skip the next step.

- **(optional)** You can also create a new environment in the anaconda
navigator and install the package there. Installing the package without creating a new environment 
(or in a fresh environment) is possible as

        conda install -c haydnspass deepsmlm
        
In python you can now import this package as simple as ```import deepsmlm```

## Setup for Developers
0. Clone the repository
1. Install conda environment from file and activate it. Use the respective environment depending on whether you have a CUDA GPU or not.

        # CUDA
        conda env create -f deepsmlm_cuda101_py38_pt14.yml
        conda activate deepsmlm_cuda   
        
        # CPU / macOS
        conda env create -f deepsmlm_cpu_py38_pt14.yml
        conda activate deepsmlm_cpu

3. Install necessary libraries or install as package.
    - **(option A)** Build libraries.
    If you have a CUDA GPU and all the CUDA stuff you need to compile something, make sure CMake can find the compiler.
    You can provide help by doing something like ```export CUDACXX=/usr/local/cuda/bin/nvcc``` but it is optional.
    If CMake does not find a working CUDA compiler it will fall back to the CPU only version.
    
            # go to cpp_src, assuming you are in the repository folder
            cd deepsmlm/cpp_src
            mkdir build
            cmake -GNinja ..
            ninja
    
    - **(option B)** Install the package. 
    This makes sure that all necessary libraries are built and installed in the right place.
    It also puts the source code in the ```.../site-packages``` folder where python expects them.

            python setup.py install
            
4. Test whether everything works as expected

            # assuming you are in the repository
            pytest deepsmlm/test
    
5. The package can be used in python as

    ```import deepsmlm```