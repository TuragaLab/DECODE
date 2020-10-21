# Tutorial
If you are planning to use the Python standalone of DECODE (i.e. this package) going through this document is 10 minutes well worthy.

# Workflow
The repeating pattern of working with this package is
1. Bead Calibration and extraction of spline coefficients (e.g. in SMAP)
2. Set training parameters by a pre-fitting procedure or reasonable guess.
3. Training of the Model
4. Fitting of the experimental data

### Bead calibration with SMAP
1. Install the stand-alone version of SMAP from [www.rieslab.de](www.rieslab.de) or if you have Matlab, downlowd the source-code from [www.github.com/jries/SMAP](www.github.com/jries/SMAP). There, you also find the installation instructions and Documentation.
2. Acquire z-stacks with fluorescent beads (e.g. 100 nm beads). We typcally use a z-range of +/- 750 nm and a step size of 10-50 nm.
3. In SMAP, use the plugin *calibrate3DSplinePSF* to generate the calibartion file. In the user guide (accessible from the SMAP help menu) in section 5.4, this is explained in detail. Further information about the calibration process can be found in [Li et al, Nature Methods (2018)](https://doi.org/10.1038/nmeth.4661).

### Determine training parameters with SMAP
1. Use the bead calibration to fit your SMLM data.
2. Use the plugin: *DECODE\_training\_estimates* to estimate the photo-physical parameters of the experiment and to save them into a parameter file. Consult the information of the plugin (accessible via the Info button) for further information.

### Training and Fit
Please load the notebooks and start up JupyterLab by executing the following in the Terminal/Anaconda Prompt:

```bash
conda activate decode_env

# load the example notebooks
python -m decode.utils.notebooks [Destination Path of the Notebooks, e.g. /Users/MaxMustermann/Downloads]  # only needed once

# fire up jupyter
jupyter lab
```
This will load the example files and open up a new brewser tab/window. Navigate to the path where you saved the notebooks and go through them.

In case of trouble please refer to the [Instructions for JupyterLab](https://jupyterlab.readthedocs.io/en/stable/getting_started/installation.html).
Note that these notebooks might change with different versions of DECODE and might be incompatible. In that case you may want to load the notebooks again (as stated above).