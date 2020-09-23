import setuptools

from setuptools import setup

# requirements when installing the wheel via pip install; conda install uses
#   info in meta.yaml instead; we're supporting multiple environments, thus
#   we have to accept some duplication (or near-duplication), unfortunately
requirements = [
    "numpy",
    "dotmap",
    "torch",
    "torchvision",
    "click",
    "deprecated",
    "dotmap",
    "h5py",
    "joblib",
    "matplotlib",
    "pandas",
    "pytest",
    "pyyaml",
    "requests",
    "scipy",
    "seaborn",
    "scikit-image",
    "scikit-learn",
    "tensorboard",
    "tifffile",
    "tqdm",
    ]


setup(
    name='decode',
    version='0.1.dev0',
    packages=setuptools.find_packages(),
    include_package_data=True,
    install_requires=requirements,
    zip_safe=False,
    url='https://rieslab.de',
    license='',
    author='Lucas-Raphael Mueller',
    author_email='',
    description=''
)
