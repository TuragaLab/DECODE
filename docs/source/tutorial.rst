Tutorial
========

If you are planning to use the Python standalone of DECODE (i.e. this package)
going through this document is 10 minutes well worthy.

Workflow
--------

The repeating pattern of working with this package is

1. Bead calibration and extraction of spline coefficients (e.g. in SMAP)
2. Set training parameters by a pre-fitting procedure or reasonable guess.
3. Training of the Model
4. Fitting of the experimental data

Bead calibration with SMAP
^^^^^^^^^^^^^^^^^^^^^^^^^^

1. Install the stand-alone version of SMAP from
   `rieslab.de <https://rieslab.de/#software>`__ or if you have MATLAB, download
   the source-code from `GitHub.com/jries/SMAP <https://github.com/jries/SMAP>`__.
   There, you also find the installation instructions and the documentation.
2. Acquire z-stacks with fluorescent beads (e.g. 100 nm beads). We typically use
   a z-range of +/- 750 nm and a step size of 10-50 nm.
3. In SMAP, use the plugin *Analyze / calibrate3DSplinePSF* to generate the
   calibration file. The plugin can be found either via tabs *Analyze / sr3D /
   calibrate3DsplinePSF* or menu *Plugins / Analyze / sr3D / calibrate3DsplinePSF*.
   More details about the process can be found in the `User Guide of SMAP
   <https://www.embl.de/download/ries/Documentation/SMAP_UserGuide.pdf#page=9>`__
   in section 5.4, in the `Step by Step Guide for SMAP
   <https://www.embl.de/download/ries/Documentation/Example_SMAP_Step_by_step.pdf#page=2>`__,
   and in the original publication `Li et al., Nature Methods (2018)
   <https://doi.org/10.1038/nmeth.4661>`__. Even for two-dimensional data you
   need a bead calibration, in this case make sure to make the `bi dir`ectional
   fit.

Determine training parameters with SMAP
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

1. Use the bead calibration to fit your SMLM data. Detailed instructions can be
   found in the `SMAP user guide
   <https://www.embl.de/download/ries/Documentation/SMAP_UserGuide.pdf#page=6>`__.
2. Use the plugin: *DECODE\_training\_estimates* to estimate the photo-physical
   parameters of the experiment and to save them into a parameter file. Consult the
   information of the plugin (accessible via the Info button) for further information.

Training and Fit
^^^^^^^^^^^^^^^^

Please load the notebooks and start up JupyterLab by executing the following in
the Terminal/Anaconda Prompt:

.. code:: bash

    conda activate decode_env

    # load the example notebooks
    python -m decode.utils.notebooks [Destination Path of the Notebooks, e.g. /Users/MaxMustermann/Downloads]  # only needed once

    # fire up jupyter
    jupyter lab

This will load the example files and open up a new browser tab/window. Navigate
to the path where you saved the notebooks and go through them.

In case of trouble please refer to the `Instructions for JupyterLab
<https://jupyterlab.readthedocs.io/en/stable/getting_started/installation.html>`__.
Note that these notebooks might change with different versions of DECODE and
might be incompatible. In that case you may want to load the notebooks again
(as stated above).

Video Tutorial
--------------

As part of the virtual `I2K 2020
<https://www.janelia.org/you-janelia/conferences/from-images-to-knowledge-with-imagej-friends>`__
conference we organized a workshop on DECODE. Please find the video below.
*DECODE is being actively developed, therefore the exact commands might differ
from those shown in the video.*

.. raw:: html

   <p style="text-align:center"><iframe width="560" height="315" src="https://www.youtube-nocookie.com/embed/zoWsj3FCUJs" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe></p>
