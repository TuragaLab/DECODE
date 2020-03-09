# DEEPsmlm
[![Build Status](https://travis-ci.com/Haydnspass/DeepSMLM.svg?token=qb4PpCab8Gb7CDLAuNTY&branch=master)](https://travis-ci.com/Haydnspass/DeepSMLM)
[![Build Status](https://travis-ci.com/Haydnspass/DeepSMLM.svg?token=qb4PpCab8Gb7CDLAuNTY&branch=dev_decode_repr)](https://travis-ci.com/Haydnspass/DeepSMLM)

## Setup
1. Install conda environment from file. For systems with a CUDA GPU use environment_cuda101_py38_pt14.yml, if no CUDA GPU is present use environment_cpu_py38_pt14.yml.
2. Activate the respective conda environment

  ```conda activate deepsmlm_cuda  # for CUDA```

  ```conda activate deepsmlm_cpu  # for CPU```

  ```source activate deepsmlm_cpu  # on macOS```  


3. Install the package
  ```python setup.py install```

4. The package can be used in python as
  ```import deepsmlm```
