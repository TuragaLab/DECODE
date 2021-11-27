"""Setup for wheel distribution. We only use this for colab. For everything else we use conda."""

import os
import setuptools

from setuptools import setup

# requirements when building the wheel via pip; conda install uses
#   info in meta.yaml instead; we're supporting multiple environments, thus
#   we have to accept some duplication (or near-duplication), unfortunately;
#   however, if conda sees the requirements here, it will be unhappy
if "CONDA_BUILD" in os.environ:
    # conda requirements set in meta.yaml
    requirements = []
else:
    # pip needs requirements here; keep in sync with meta.yaml!
    requirements = [
        "numpy",
        # HACK: We omit torch version to be as flexible to the version
        # as we can so that slight changes on colab do not break so fast
        # that's why we deviate from enviornment.yaml
        "torch",
        "click",
        "deprecated",
        "gitpython>=3.1",
        "h5py",
        "importlib_resources",
        "matplotlib",
        "pandas",
        "pytest",
        "pyyaml",
        "requests",
        "scipy",
        "seaborn==0.10",
        "scikit-image",
        "scikit-learn",
        "tensorboard",
        "tifffile",
        "tqdm",
        ]

setup(
    name='decode',
    version='0.10.1dev1',  # do not modify by hand set and sync with bumpversion
    packages=setuptools.find_packages(),
    include_package_data=True,
    install_requires=requirements,
    entry_points={
        'console_scripts': [
            'decode.train = decode.neuralfitter.train.train:main',
            'decode.fit = decode.neuralfitter.inference.infer:main',
            'decode.infer = decode.neuralfitter.inference.infer:main',
        ],
    },
    zip_safe=False,
    url='https://rieslab.de',
    license='GPL3',
    author='Lucas-Raphael Mueller',
    author_email='',
    description=''
)
