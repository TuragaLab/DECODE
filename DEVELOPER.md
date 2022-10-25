## Instructions for Developers

0. Clone the repository
1. Install conda environment from file and activate it. Use the respective environment depending on whether you have a CUDA GPU or not.
The cubic spline psf is pre-compiled in a different repo / as a separate package and will be installed automatically.

        # for CUDA (change cpuonly to cudatoolkit in environment.yaml)
        conda env create -f environment.yaml

3. Test whether everything works as expected. Note that if you run all tests, all files from the gateway file will be downloaded but you can exclude these tests.

            # assuming you are in the repo folder
            
            pytest test  # all tests
            pytest -m "not webbig"  # without downloading the assets
            pytest -m "not plot" test  # without the tests that have graphical output

4. The package can be used in python as

    ```import decode```

### Note
Simulation heavily relies on the Cubic Spline PSF Implementation (Li, Y. et al. Real-time 3D single-molecule localization using experimental point spread functions. Nature Methods 15, 367–369 (2018)).
It has been factored out in a seperate Repo to make life easier (see https://github.com/Haydnspass/SplinePSF). It'll be auto-installed.

### Building the Docs
Install environment as described above
```bash
cd docs
make html
```
The docs can be found in the build folder.
 

### Building and Deploying
Please follow the instructions as described here. Note that create the wheels only for Google Colab. 
All users should install via conda.
```bash
# optional, only once: create a new conda build environment
conda create --name build_clean conda-build anaconda bump2version -c conda-forge
conda activate build_clean

# bump version so that all versions get updated automatically, creates a git version tag automatically
bump2version [major/minor/patch/release/build]  # --verbose --dry-run to see the effect

# upload git tag
git push --tags

# build wheels
python setup.py bdist_wheel
# edit git release and upload the wheels

# conda release
cd conda
conda-build -c turagalab -c pytorch -c conda-forge decode

anaconda upload -u [your username] [path as provided at the end of the conda-build output]
```
After this you may test the build in a clean environment.
You may update the link to the wheels in the `gateway.yaml` file in order to let Colab download the respective version.
As we want to have this tested first, we did not automate this.


## Gateway
Notebooks and Colab depend on `gateway.yaml`. The only relevant version is the one in the master 
branch. That means, in master you need to keep all packages that are potentially used.

```bash
# hashing a new .zip package
shasum -a 256 [.zip file]

```

## Hdf5 structure
The h5 structure of decode EmitterSets is as follows (example shows a random emitter):
```bash
{'data': {
    'bg': array([nan, nan, nan], dtype=float32),
    'bg_cr': None,
    'bg_sig': None,
    'frame_ix': array([0, 0, 0]),
    'id': array([-1, -1, -1]),
    'phot': array([1., 1., 1.], dtype=float32),
    'phot_cr': None,
    'phot_sig': None,
    'prob': array([1., 1., 1.], dtype=float32),
    'xyz': array([[23.236893 , 13.064125 , 25.711937 ],
                  [ 9.199045 ,  7.3632336, 28.120571 ],
                  [21.179623 , 25.227995 , 25.280546 ]], dtype=float32),
    'xyz_cr': None,
    'xyz_sig': None
    },
 'decode': {
    'version': '0.10.0'
    },
 'meta': {
   'px_size': array([100., 100.], dtype=float32), 
   'xy_unit': 'px'
   }
}
```