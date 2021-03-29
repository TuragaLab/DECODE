Logging
=======

Currently we support monitoring the training progress in Tensorboard while basic
metrics are reported to the console as well. All metrics that include comparison
to ground truth emitters are based on the parameters (implicitly) provided in
the configuration .yaml file. Those include match dimensionality (i.e., in 2D or
3D) and max. allowed distances. The threshold on the detection filters the
detections prior to matching.

Tensorboard
-----------

Metrics
^^^^^^^

+----------------+-------------------------+------------------------------------------------------------------+
| Abbreviation   | Name                    | Description                                                      |
+================+=========================+==================================================================+
| prec           | Precision               | Number of true positives over all detections                     |
+----------------+-------------------------+------------------------------------------------------------------+
| rec            | Recall                  | Number of true positives over all (ground truth) localizations   |
+----------------+-------------------------+------------------------------------------------------------------+
| jac            | Jaccard                 | Intersection over Union                                          |
+----------------+-------------------------+------------------------------------------------------------------+
| f1             | F1 Score                | 2 x Precision x Recall over (Precision + Recall)                 |
+----------------+-------------------------+------------------------------------------------------------------+
| effcy_lat      | Efficiency lateral      | SMLM Challenge defined efficiency lateral                        |
+----------------+-------------------------+------------------------------------------------------------------+
| effcy_ax       | Efficiency axial        | SMLM Challenge defined efficiency axial                          |
+----------------+-------------------------+------------------------------------------------------------------+
| effcy_vol      | Efficiency volumetric   | SMLM Challenge defined efficiency volumetric                     |
+----------------+-------------------------+------------------------------------------------------------------+
| rmse_lat       | RMSE lateral            | Root Mean Square Error lateral                                   |
+----------------+-------------------------+------------------------------------------------------------------+
| rmse_ax        | RMSE axial              | Root Mean Square Error axial (z)                                 |
+----------------+-------------------------+------------------------------------------------------------------+
| rmse_vol       | RMSE volumetric         | Root Mean Square Error volumetric                                |
+----------------+-------------------------+------------------------------------------------------------------+
| mad_lat        | MAD lateral             | Mean Absolute Deviation lateral                                  |
+----------------+-------------------------+------------------------------------------------------------------+
| mad_ax         | MAD axial               | Mean Absolute Deviation axial                                    |
+----------------+-------------------------+------------------------------------------------------------------+
| mad_vol        | MAD volumetric          | Mean Absolute Deviation volumetric                               |
+----------------+-------------------------+------------------------------------------------------------------+

Other (non or less-semantic) metrics

+------------------+------------------------------+
| Abbreviation     | Description                  |
+==================+==============================+
| learning_rate    | Learning Rate                |
+------------------+------------------------------+
| test_ep          | Test loss per epoch          |
+------------------+------------------------------+
| train_batch      | Train loss per batch         |
+------------------+------------------------------+
| train_ep         | Train loss per epoch         |
+------------------+------------------------------+
| loss_cmp_0       | Gaussian localization loss   |
+------------------+------------------------------+
| loss_cmp_1       | Background loss              |
+------------------+------------------------------+

Distributions
^^^^^^^^^^^^^

+----------------------------+-------------------------------------------------------------------------------------+
| Abbreviation               | Description                                                                         |
+============================+=====================================================================================+
| dist/prob                  | Probability distribution of active pixels (active typically means above p >= 0.6)   |
+----------------------------+-------------------------------------------------------------------------------------+
| dist/x_offset              | Distribution of relative offset to pixel border in x direction                      |
+----------------------------+-------------------------------------------------------------------------------------+
| dist/y_offset              | Distribution of relative offset to pixel border in y direction                      |
+----------------------------+-------------------------------------------------------------------------------------+
| residuals/phot_gt_pred     | Predicted vs ground truth photon counts                                             |
+----------------------------+-------------------------------------------------------------------------------------+
| residuals/z_gt_pred        | Predicted vs ground truth z values                                                  |
+----------------------------+-------------------------------------------------------------------------------------+

In- Output Logging
^^^^^^^^^^^^^^^^^^

Input shows a sampled input frame with ground truth emitters. Output shows the
respective output channels of the model.

Emitter-Out and Emitter-Match
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

+-----------------+--------------------------------------------------------------+
| Abbreviation    | Description                                                  |
+=================+==============================================================+
| em_match        | All matched emitter pairs                                    |
+-----------------+--------------------------------------------------------------+
| em_match_3d     | All matched emitter pairs in 3D                              |
+-----------------+--------------------------------------------------------------+
| em_out          | All ground truth emitters and all predicted emitters         |
+-----------------+--------------------------------------------------------------+
| em_out_3d       | All ground truth emitters and all predicted emitters in 3D   |
+-----------------+--------------------------------------------------------------+
