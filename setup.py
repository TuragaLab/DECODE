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
        "torch==1.6",
        "torchvision",
        "click",
        "deprecated",
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
    version='0.9.4',  # do not modify by hand set and sync with bumpversion
    packages=setuptools.find_packages(),
    include_package_data=True,
    install_requires=requirements,
    zip_safe=False,
    url='https://rieslab.de',
    license='GPL3',
    author='Lucas-Raphael Mueller',
    author_email='',
    description=''
)
