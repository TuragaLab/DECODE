============
Data
============

We provide experimental data for you to try out DECODE. If you want to go through the whole
pipeline, i.e. including your own bead calibration and training parametrization
(i.e. :ref:`bead calibration and prefit; steps 1 and 2 <Workflow>`) you can find the URLs to
download example data from our
`gateway <https://github.com/TuragaLab/DECODE/blob/master/gateway.yaml>`__.
If you want to omit these steps and try out DECODE directly, the
data will be downloaded automatically in the respective Jupyter Notebooks.

Note that we have taken quite some measures to ensure the availability of the example data.
If you still encounter a dead link, please notify us by
`opening an issue <https://github.com/TuragaLab/DECODE/issues>`__.

DECODE on your own data
=======================

If you want to fit your own data, there are few small points you need to be aware of:

-   Make sure to set the correct camera parameters in SMAP when performing the prefitting routine.
    In particular, there is the setting "Mirror" for which we have a dedicated note in the
    Fitting notebook (please see there).

-   If your frame size is not a multiple of 8 (for the default DECODE) model, the frame will be
    *centre-cropped*. This is due to the intrinsics of the CNN. You are free to use padding to
    multiples of 8 instead of cropping but this could lead to distortion effects at the frame border.


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
