# DEEPsmlm
[![Build Status](https://travis-ci.com/Haydnspass/DeepSMLM.svg?token=qb4PpCab8Gb7CDLAuNTY&branch=master)](https://travis-ci.com/Haydnspass/DeepSMLM)

### Notes
Simulation heavily relies on the Cubic Spline PSF Implementation (Li, Y. et al. Real-time 3D single-molecule localization using experimental point spread functions. Nature Methods 15, 367–369 (2018)).
It has been factored out in a seperate Repo to make life easier (see https://github.com/Haydnspass/SplinePSF).


## Installation for Users
The easiest way to get it, is by using conda. If you don't have conda (chances are you have it when you have used python)
you may download it from https://anaconda.org.

In the following we will make a conda environment and install deepsmlm. If you are familiar with using
```conda``` in your terminal of choice that's great! Just type:

        conda create -n deepsmlm_env -c haydnspass -c pytorch -c conda-forge deepsmlm python=3.8  # python >=3.6 supported
        conda activate deepsmlm_env
        
This will create a new environment and install this framework. 
If everything worked until here you are good and can skip the next step.

- **(optional)** You can also create a new environment in the anaconda
navigator and install the package there. Installing the package without creating a new environment 
(or in a fresh environment) is possible as

        conda install -c haydnspass -c pytorch deepsmlm
        
In python you can now import this package as simple as ```import deepsmlm```

## Setup for Developers
0. Clone the repository
1. Install conda environment from file and activate it. Use the respective environment depending on whether you have a CUDA GPU or not.
The cubic spline psf is pre-compiled in a different repo / as a separate package and will be installed automatically.

        # CUDA
        conda env create -f deepsmlm_cuda101_py38_pt14.yml
        conda activate deepsmlm_cuda   
        
        # CPU / macOS
        conda env create -f deepsmlm_cpu_py38_pt14.yml
        conda activate deepsmlm_cpu
            
3. Test whether everything works as expected

            # assuming you are in the repo folder
            pytest deepsmlm/test
            
            # or if you fancy some nice figures, depending on your IDE 
            # you might need to close popping up matplot figures
            pytest deepsmlm/test --plot  
    
4. The package can be used in python as

    ```import deepsmlm```