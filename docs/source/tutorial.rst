========
Tutorial
========

Here we describe how to use DECODE locally, i.e. when you want to use it on a regular basis.
If you want to test DECODE without installation you can check out the Google Colab notebooks
linked on the `starting page <index.html#google-colab-notebooks>`__ of this documentation.

**Note:** This tutorial assumes that you have successfully installed DECODE locally and got your
copy of the DECODE jupyter notebooks. If this is not the case for you, please refer to the
`installation instructions <installation.html>`__ and follow the step-by-step guide.


.. _Workflow:

********
Workflow
********

A typical workflow for fitting high-density SMLM data with this package is

1. :ref:`Bead calibration <Bead calibration>` and extraction of spline coefficients (e.g. in SMAP)
2. :ref:`Set training parameters <Training parameters>` by a pre-fitting procedure or reasonableguess.
3. :ref:`Training a DECODE model <Training>`
4. :ref:`Fitting experimental data <Fit>`
5. :ref:`Export, visualization and analysis <Visualization>` of fitted data

The first two steps involving SMAP can be skipped and you can start right away
with the :ref:`notebooks <First time>` in case you want to work with our
example data, as we provide the intermediate result files (i.e. the calibration and the training
parametrization). If you are working with your own data or want to go through the whole workflow,
just start from the beginning.


.. _Bead calibration:

Bead calibration with SMAP
==========================

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
=======================================

1. Use the bead calibration to fit your SMLM data. Detailed instructions can be
   found in the `SMAP user guide
   <https://www.embl.de/download/ries/Documentation/SMAP_UserGuide.pdf#page=6>`__.
2. Use the plugin: *DECODE\_training\_estimates* to estimate the photo-physical
   parameters of the experiment and to save them into a parameter file. Consult the
   information of the plugin (accessible via the Info button) for further information.


.. _Training:

Training a DECODE model
=======================

The basis for training DECODE is a parametrization of training procedure. This parametrization is
described in a simple `.yaml` file which holds a couple of paths (e.g. the calibration file and
your output directory) as well as the parametrization of the simulation that should somewhat
match the data you want to fit.

In our Training notebook we guide you through the process of creating such a `.yaml` file that can
subsequently used to start the actual training.

If you have gone through the notebooks already and generated your own `param.yaml` file, you can skip
the following section and go to the :ref:`regular workflow <Regular workflow>` directly.

.. _First time:

First time Using DECODE
-----------------------

To get you up and running, we provide several notebooks that introduce DECODE to you.
In total, there are four different notebooks:

- **Introduction** contains a general introduction to DECODE and helps you to get familiar with the framework.
- **Training** guides you through creating a parameter file that you need for training the model. It is based on emitter characteristics determined previously (or provided by the example).
- **Fitting** localizes the single molecules in the high-density data based on the model.
- **Evaluation** gives you an introduction to the post-processing capabilities of DECODE.

To start going through the notebooks execute the following command in your Terminal/Anaconda Prompt:

.. code:: bash

    # fire up jupyter
    jupyter lab

This will load the example files and opens up a new browser tab/window. Navigate
to the path where you saved the notebooks and go through them. We recommend to start with the
Introduction notebook, followed by Training and Fitting.

.. _Regular workflow:

Regular workflow
----------------

In practice you can either write such a `.yaml` file directly, i.e. by educated guessing your
emitter characteristics, or follow the pre-fit routine using SMAP that will auto-generate it.

Once being equipped with your calibration and the parameter file, you can start the training in
your Terminal/Anaconda prompt

.. code:: bash

    conda activate decode_env
    python -m decode.neuralfitter.train.live_engine -p [path to your param].yaml



.. _Fit:

Fit
===

Please refer to the :ref:`First Time using DECODE instructions. <First time>`


.. _Visualization:

Export from DECODE and import in SMAP for visualization
=======================================================

DECODE has basic rendering functions but for detailed visualization and analysis you should export
your data and load it into SMAP or another SMLM visualization software of your choice.

For loading the data in SMAP, you can export your emitter set as h5 file at the end of the fitting notebook.
For easier input in other software we recommend exporting as csv.
Under the *File* tab in SMAP, change the *auto loader* to *Import DECODE .csv/.h5* and **Load** the exported data.
For detailed instructions on post-processing (grouping, filtering, drift correction,...)
please consult the `SMAP Documentation <https://www.embl.de/download/ries/Documentation/>`__,
more specifically from point 5 onwards in the
`Getting Started Guide <https://www.embl.de/download/ries/Documentation/Getting_Started.pdf#page=4>`__
and from point 6 on in the
`SMAP User Guide <https://www.embl.de/download/ries/Documentation/SMAP_UserGuide.pdf#page=11>`__.


.. _Example Data:

************
Example Data
************

We provide experimental data for you to try out DECODE. If you want to do the whole pipeline, i.e.
including your own bead calibration and training parametrization
(i.e. :ref:`steps 1 and 2 <Workflow>`) you can download example data from our
`gateway <https://github.com/TuragaLab/DECODE/blob/master/gateway.yaml>`__.
If you want to omit the :ref:`steps 1 and 2 <Workflow>` and try out DECODE directly, the
data will be downloaded automatically in the respective Jupyter Notebooks.

Note that we have taken quite some measures to ensure the availability of the example data.
If you still encounter a dead link, please notify us by
`opening an issue <https://github.com/TuragaLab/DECODE/issues>`__.

Experimental data
=================

We provide the RAW data, RAW beads, training parametrization and converged model to reproduce
Figure 4 of our preprint. The notebooks automatically download this package.
For manual download the link can be found in our
`gateway <https://github.com/TuragaLab/DECODE/blob/master/gateway.yaml>`__
(experimental_data_workflow).

SMLM Challenge data
===================

If you want to reproduce our results for the SMLM challenge, you can download the respective
datasets on the `SMLM Challenge Website <http://bigwww.epfl.ch/smlm/datasets/index.html>`__.
Other than that you can follow the exact same steps as described in the workflow above.
You can find our parameter files for the challenge in our
`gateway <https://github.com/TuragaLab/DECODE/blob/master/gateway.yaml>`__
(smlm_challenge package).

The camera parameters are as follows:

Camera Parameters
"""""""""""""""""

+---------------------+-------------+-------------+
|                     | 3D AS       | 3D DH       |
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
