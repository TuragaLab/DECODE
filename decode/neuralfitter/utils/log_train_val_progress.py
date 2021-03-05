import warnings

import matplotlib.pyplot as plt
import torch
import seaborn as sns

import decode.generic.emitter
from decode.evaluation.evaluation import WeightedErrors
from decode.evaluation import predict_dist
from decode.plot import frame_coord

from decode.evaluation import evaluation


def log_frames(x, y_out, y_tar, weight, em_out, em_tar, tp, tp_match, logger, step, colorbar=True):

    r_ix = torch.randint(0, len(x), (1, )).long().item()
    assert x.dim() == 4

    # rm batch dimension, i.e. select one sample
    x = x[r_ix]
    y_out = y_out[r_ix]
    y_tar = y_tar[r_ix] if y_tar is not None else None
    weight = weight[r_ix] if weight is not None else None

    assert isinstance(em_tar, decode.generic.emitter.EmitterSet)
    em_tar = em_tar.get_subset_frame(r_ix, r_ix)
    em_out = em_out.get_subset_frame(r_ix, r_ix)
    em_tp = tp.get_subset_frame(r_ix, r_ix)
    em_tp_match = tp_match.get_subset_frame(r_ix, r_ix)

    # loop over all input channels
    for i, xc in enumerate(x):
        f_input = plt.figure()
        frame_coord.PlotFrameCoord(xc, pos_tar=em_tar.xyz_px, plot_colorbar_frame=colorbar).plot()
        logger.add_figure('input/raw_input_ch_' + str(i), f_input, step)

    # loop over all output channels
    for i, yc in enumerate(y_out):
        f_out = plt.figure()
        frame_coord.PlotFrameCoord(yc, plot_colorbar_frame=colorbar).plot()
        logger.add_figure('output/raw_output_ch_' + str(i), f_out, step)

    # record tar / output emitters
    tar_ch = (x.size(0) - 1) // 2

    f_em_out = plt.figure(figsize=(10, 8))
    frame_coord.PlotFrameCoord(x[tar_ch], pos_tar=em_tar.xyz_px, pos_out=em_out.xyz_px).plot()
    logger.add_figure('em_out/em_out_tar', f_em_out, step)

    f_em_out3d = plt.figure(figsize=(10, 8))
    frame_coord.PlotCoordinates3D(pos_tar=em_tar.xyz_px, pos_out=em_out.xyz_px).plot()
    logger.add_figure('em_out/em_out_tar_3d', f_em_out3d, step)

    f_match = plt.figure(figsize=(10, 8))
    frame_coord.PlotFrameCoord(x[tar_ch], pos_tar=em_tp_match.xyz_px, pos_out=em_tp.xyz_px, match_lines=True,
                               labels=('TP match', 'TP')).plot()
    logger.add_figure('em_out/em_match', f_match, step)

    f_match_3d = plt.figure(figsize=(10, 8))
    frame_coord.PlotCoordinates3D(pos_tar=em_tp_match.xyz_px, pos_out=em_tp.xyz_px, match_lines=True,
                                  labels=('TP match', 'TP')).plot()
    logger.add_figure('em_out/em_match_3d', f_match_3d, step)

    # loop over all target channels
    if y_tar is not None:
        for i, yct in enumerate(y_tar):
            f_tar = plt.figure()
            frame_coord.PlotFrameCoord(yct, plot_colorbar_frame=colorbar).plot()
            logger.add_figure('target/target_ch_' + str(i), f_tar, step)

    # loop over all weight channels
    if weight is not None:
        for i, w in enumerate(weight):
            f_w = plt.figure()
            frame_coord.PlotFrameCoord(w, plot_colorbar_frame=colorbar).plot()
            logger.add_figure('weight/weight_ch_' + str(i), f_w, step)

    # plot dist of probability channel
    # ToDo: Histplots seem to cause trouble with memory. Deactivated for now. If reactivate: change back to distplot
    # f_prob_dist, ax_prob_dist = plt.subplots()
    # sns.histplot(y_out[0].reshape(-1).numpy(), kde=False, ax=ax_prob_dist)
    # plt.xlabel('prob')
    # logger.add_figure('output_dist/prob', f_prob_dist)
    #
    # f_prob_dist_log, ax_prob_dist_log = plt.subplots()
    # sns.histplot(y_out[0].reshape(-1).numpy(), kde=False, ax=ax_prob_dist_log)
    # plt.yscale('log')
    # plt.xlabel('prob')
    # logger.add_figure('output_dist/prob_log', f_prob_dist_log)


def log_kpi(loss_scalar: float, loss_cmp: dict, eval_set: dict, logger, step):

    logger.add_scalar('learning/test_ep', loss_scalar, step)

    assert loss_cmp.dim() >= 2
    for i in range(loss_cmp.size(1)):  # channel-wise mean
        logger.add_scalar('loss_cmp/test_ep_loss_ch_' + str(i), loss_cmp[:, i].mean(), step)

    logger.add_scalar_dict('eval/', eval_set, step)


def log_dists(tp, tp_match, pred, px_border, px_size, logger, step):

    """Log z vs z_gt"""
    f_x, ax_x = plt.subplots()
    f_y, ax_y = plt.subplots()
    f_z, ax_z = plt.subplots()
    f_phot, ax_phot = plt.subplots()

    predict_dist.emitter_deviations(tp, tp_match,
                                    px_border=px_border, px_size=px_size, axes=[ax_x, ax_y, ax_z, ax_phot])

    logger.add_figure('dist/x_offset', f_x, step)
    logger.add_figure('dist/y_offset', f_y, step)
    logger.add_figure('residuals/z_gt_pred', f_z, step)
    logger.add_figure('residuals/phot_gt_pred', f_phot, step)

    """Log prob dist"""
    f_prob, ax_prob = plt.subplots()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sns.distplot(pred.prob, bins=50, norm_hist=True, ax=ax_prob, kde=False)
    logger.add_figure('dist/prob', f_prob, step)


def log_train(*, loss_p_batch: (list, tuple), loss_mean: float, logger, step: int):

    logger.add_scalar('learning/train_ep', loss_mean, step)

    for i, loss_batch in enumerate(loss_p_batch):
        step_batch = step * len(loss_p_batch) + i
        if i % 10 != 0:
            continue

        logger.add_scalar('learning/train_batch', loss_batch, step_batch)


def post_process_log_test(*, loss_cmp, loss_scalar, x, y_out, y_tar, weight, em_tar,
                          px_border, px_size, post_processor, matcher, logger, step):

    """Post-Process"""
    em_out = post_processor.forward(y_out)

    """Match and Evaluate"""
    tp, fp, fn, tp_match = matcher.forward(em_out, em_tar)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = evaluation.SMLMEvaluation(weighted_eval=WeightedErrors(mode='crlb', reduction='gaussian')).forward(tp, fp, fn, tp_match)

    """Log"""
    # raw frames
    log_frames(x=x, y_out=y_out, y_tar=y_tar, weight=weight, em_out=em_out, em_tar=em_tar, tp=tp, tp_match=tp_match,
               logger=logger, step=step)

    # KPIs
    log_kpi(loss_scalar=loss_scalar, loss_cmp=loss_cmp, eval_set=result._asdict(), logger=logger, step=step)

    # distributions
    log_dists(tp=tp, tp_match=tp_match, pred=em_out, px_border=px_border, px_size=px_size, logger=logger, step=step)

    return
