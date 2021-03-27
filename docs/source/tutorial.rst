Tutorial
========

Here we describe how to use DECODE locally, i.e. when you want to use it on a regular basis.
If you want to test DECODE without installation you can check out the Google Colab notebooks
linked on the `starting page <index.html#google-colab-notebooks>`__ of this documentation.

.. _Workflow:

Workflow
--------

A typical workflow for fitting high-density SMLM data with this package is

1. :ref:`Bead calibration <Bead calibration>` and extraction of spline coefficients (e.g. in SMAP)
2. :ref:`Set training parameters <Training parameters>` by a pre-fitting procedure or reasonableguess.
3. :ref:`Training <Training and Fit>` of the model
4. :ref:`Fitting <Training and Fit>` of the experimental data
5. :ref:`Export, visualization and analysis <Visualization>` of fitted data

The first two steps involving SMAP can be skipped and you can start right away
with the :ref:`notebooks <Training and Fit>` in case you want to work with our
example data, as we provide the intermediate result files (i.e. the calibration and the training
parametrization). If you are working with your own data or want to go through the whole workflow,
just start from the beginning.

.. _Bead calibration:

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
   need a bead calibration, in this case make sure to make the *bi directional
   fit*.

.. _Training parameters:

Determine training parameters with SMAP
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

1. Use the bead calibration to fit your SMLM data. Detailed instructions can be
   found in the `SMAP user guide
   <https://www.embl.de/download/ries/Documentation/SMAP_UserGuide.pdf#page=6>`__.
2. Use the plugin: *DECODE\_training\_estimates* to estimate the photo-physical
   parameters of the experiment and to save them into a parameter file. Consult the
   information of the plugin (accessible via the Info button) for further information.

.. _Training and Fit:

Training and Fit
^^^^^^^^^^^^^^^^

The training and fit are performed in Jupyter notebooks.
Please load the notebooks and start up JupyterLab by executing the following in the
Terminal/Anaconda Prompt.

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

There are four different notebooks:

- **Introduction** contains a general introduction to DECODE and helps you to get familiar with the framework.
- **Training** guides you through the model training based on the emitter characteristics determined previously (or provided by the example).
- **Fitting** localizes the single molecules in the high-density data based on the model.
- **Evaluation** gives you an introduction to the post-processing capabilities of DECODE.

The Training and Fitting notebooks are also available on Google Colab to test
DECODE without installation. They contain example data and step by step guides.
The most up-to-date notebooks are linked on the
`README <https://github.com/TuragaLab/DECODE>`__ of the DECODE GitHub
repository. They have been tested with our example data but due to limited
hardware we cannot guarantee that they also work with your own raw data.

.. _Visualization:

Export from DECODE and import in SMAP for visualization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
DECODE has basic rendering functions but for detailed visualization and analysis you should export your data and load it into SMAP or another SMLM visualization software of your choice.

For loading the data in SMAP, you can export your emitter set as h5 file at the end of the fitting notebook. For easier input in other software we recommend exporting as csv.
Under the *File* tab in SMAP, change the *auto loader* to *Import DECODE .csv/.h5* and **Load** the exported data. For detailed instructions on post-processing (grouping, filtering, drift correction,...) please consult the `SMAP Documentation <https://www.embl.de/download/ries/Documentation/>`__, more specifically from point 5 onwards in the `Getting Started Guide <https://www.embl.de/download/ries/Documentation/Getting_Started.pdf#page=4>`__ and from point 6 on in the `SMAP User Guide <https://www.embl.de/download/ries/Documentation/SMAP_UserGuide.pdf#page=11>`__.


.. _Example Data:

Example Data
------------

We provide experimental data for you to try out DECODE. If you want to do the whole pipeline, i.e.
including your own bead calibration and training parametrization
(i.e. :ref:`steps 1 and 2 <Workflow>`) you can download the example data from our
`gateway <https://github.com/TuragaLab/DECODE/blob/master/gateway.yaml>`__.
If you want to omit the :ref:`steps 1 and 2 <Workflow>` and try out DECODE directly, the training
data will be downloaded automatically in the respective Jupyter Notebooks.

Note that we have taken quite some measures to ensure the availability of the example data. If
you still encounter a dead link, please notify us by
`opening an issue <https://github.com/TuragaLab/DECODE/issues>`__.


SMLM Challenge data
^^^^^^^^^^^^^^^^^^^
If you want to reproduce our results for the SMLM challenge, you can download the respective
datasets on the `SMLM Challenge Website <http://bigwww.epfl.ch/smlm/datasets/index.html>`__.
Other than that you can follow the exact same steps as described in the workflow above.
To get you started right away, please note the following:

Camera Parameters
"""""""""""""""""

+---------------------+-------------+-------------+
| Parameters\Modality | 3D AS       | 3D DH       |
+---------------------+-------------+-------------+
| baseline            | 100.0       | 100.0       |
+---------------------+-------------+-------------+
| e_per_adu           | 45.0        | 45.0        |
+---------------------+-------------+-------------+
| em_gain             | 300         | 300         |
+---------------------+-------------+-------------+
| qe                  | 1. :sup:`†` | 1. :sup:`†` |
+---------------------+-------------+-------------+
| read_sigma          | 74.4        | 74.4        |
+---------------------+-------------+-------------+
| spur_noise          | 0.002       | 0.002       |
+---------------------+-------------+-------------+
| px_size             | [100, 100]  | [100, 00]   |
+---------------------+-------------+-------------+

:sup:`†` we typically use a *quantum efficiency* of 1. and refer to the photons as *detected
photons.*
For direct challenge comparison, the photon count must then be adjusted by 1/ 0.9 (where 0.9 is the
quantum efficiency of the camera for the simulated 3D AS/DH data).

Moreover, for this data *Mirroring must be turned off* both in SMAP (Camera Parameters) as well
as in the Fitting notebook (see the details there).
