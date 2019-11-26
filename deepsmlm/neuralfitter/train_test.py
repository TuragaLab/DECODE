import os
import random
import time
import tqdm

import numpy as np
from copy import deepcopy
import torch
torch.multiprocessing.set_sharing_strategy('file_system')
from matplotlib import pyplot as plt

from deepsmlm.generic.emitter import EmitterSet
from deepsmlm.generic.plotting.frame_coord import PlotFrameCoord, PlotCoordinates3D, PlotFrame
import deepsmlm.evaluation.evaluation as eval


"""Several pseudo-global variables useful for data processing and debugging."""
deepsmlm_root = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)),
                 os.pardir, os.pardir)) + '/'

DEBUG = True
LOG = True
TENSORBOARD = True
LOG_FIGURES = True

LOG = True if DEBUG else LOG


def monitor_io(input, output, em_out, target, em_tar):
    r_ix = random.randint(0, input.size(0))

    input_ = input[r_ix]
    out_ = output[r_ix]
    tar_ = target[r_ix]

    em_out_ = em_out[r_ix]
    em_tar_ = em_tar[r_ix]

    plt.figure(figsize=(20, 12))
    plt.subplot(131)
    PlotFrameCoord(input_[0], pos_tar=em_tar_.xyz).plot()
    plt.subplot(132)
    PlotFrameCoord(input_[1], pos_tar=em_tar_.xyz).plot()
    plt.subplot(133)
    PlotFrameCoord(input_[2], pos_tar=em_tar_.xyz).plot()
    plt.show()

    plt.figure(figsize=(20, 20))
    plt.subplot(231)
    PlotFrameCoord(input_[1], pos_out=em_out_.xyz).plot()
    plt.subplot(232)
    PlotFrameCoord(out_[0]).plot()
    plt.subplot(233)
    PlotFrameCoord(out_[1]).plot()
    plt.subplot(234)
    PlotFrameCoord(out_[2]).plot()
    plt.subplot(235)
    PlotFrameCoord(out_[3]).plot()
    plt.subplot(236)
    PlotFrameCoord(out_[4]).plot()
    plt.show()

    plt.figure(figsize=(20, 20))
    plt.subplot(231)
    PlotFrameCoord(input_[1], pos_tar=em_tar_.xyz).plot()
    plt.subplot(232)
    PlotFrameCoord(tar_[0]).plot()
    plt.subplot(233)
    PlotFrameCoord(tar_[1]).plot()
    plt.subplot(234)
    PlotFrameCoord(tar_[2]).plot()
    plt.subplot(235)
    PlotFrameCoord(tar_[3]).plot()
    plt.subplot(236)
    PlotFrameCoord(tar_[4]).plot()
    plt.show()

    plt.figure(figsize=(20, 20))
    PlotFrameCoord(input_[1], pos_tar=em_tar_.xyz, pos_out=em_out_.xyz).plot()
    plt.show()


def plot_io_frame_model(frame, output, target, em_tar, indices, fig_str, comet_log, board_log, step):
    channel = 0 if (frame.shape[1] == 1) else 1
    figures = []

    for i, ix in enumerate(indices):
        img = frame[ix, channel, :, :].detach().cpu()
        img_out = output[ix, 0, :, :].detach().cpu()
        xyz_target = em_tar[ix, :].detach().cpu()
        fig = plt.figure()
        plt.subplot(121)
        PlotFrameCoord(frame=img, pos_tar=xyz_target).plot()
        plt.title('Epoch {} | SampleIx {} | Channel: {}'.format(step, ix, channel))
        plt.subplot(122)
        # TODO: Extent should not be determined by img-shape but rather by specification.
        PlotFrameCoord(frame=img_out, pos_tar=xyz_target,
                       extent=((-0.5, img.shape[0] - 0.5), (-0.5, img.shape[1] - 0.5))).plot()
        figures.append(fig)
        plt.show()

        fig_str_ = fig_str + str(i)
        comet_log.log_figure(fig_str_, fig)
        board_log.add_figure(fig_str_, fig, step)


def plot_io_coord_model(frame, output, target, em_tar, indices, fig_str, comet_log, board_log, step, D3=False):
    channel = 0 if (frame.shape[1] == 1) else 1
    figures = []
    figures_3d = []
    for i, ix in enumerate(indices):
        img = frame[ix, channel, :, :].detach().cpu()
        xyz_tar = target[0][ix, :].detach().cpu()
        xyz_out = (output[0][ix, :]).detach().cpu()
        phot_tar = target[1][ix].detach().cpu()
        phot_out = output[1][ix].detach().cpu()
        fig = plt.figure()
        PlotFrameCoord(frame=img,
                       pos_tar=xyz_tar,
                       phot_tar=None,  # phot_tar,
                       pos_out=xyz_out,
                       phot_out=phot_out).plot()
        plt.title('Epoch {} | SampleIx {} | Channel: {}'.format(step, ix, channel))
        plt.legend()
        figures.append(fig)
        plt.show()

        fig_str_ = fig_str + str(i)
        comet_log.log_figure(fig_str_, fig)
        board_log.add_figure(fig_str_, fig, step)

        if D3:
            fig = plt.figure()
            PlotCoordinates3D(pos_tar=xyz_tar,
                              pos_out=xyz_out,
                              phot_out=phot_out).plot()
            plt.title('Epoch {} | Sampleix {} | Channel: {}'.format(step, ix, channel))
            plt.legend()
            figures_3d.append(fig)
            plt.show()

            fig_str_ = fig_str + str(i) + '_3D'
            comet_log.log_figure(fig_str_, fig)
            board_log.add_figure(fig_str_, fig, step)

    return figures, figures_3d


def train(train_loader, model, optimizer, criterion, epoch, conf_param, logger, experiment):
    step_batch = epoch * train_loader.__len__()

    batch_time = eval.MetricMeter()
    data_time = eval.MetricMeter()
    losses = eval.MetricMeter()
    loss_batch10 = eval.MetricMeter()

    model.train()
    end = time.time()
    train_loader_tqdm = tqdm.tqdm(train_loader, total=train_loader.__len__(), ncols=110, smoothing=0.)
    for i, (x_in, target, weights) in enumerate(train_loader_tqdm):

        # measure data loading time
        data_time.update(time.time() - end)

        x_in = x_in.to(torch.device(conf_param['Hardware']['device']))
        target = target.to(torch.device(conf_param['Hardware']['device']))
        weights = weights.to(torch.device(conf_param['Hardware']['device']))

        # compute output
        output = model(x_in)
        loss_ = criterion(output, target, weights)  # alternate batch_wise

        loss = loss_.mean()
        # record loss
        losses.update(loss.item())
        loss_batch10.update(loss.item())

        # compute gradients in a backward pass
        optimizer.zero_grad()
        loss.backward()

        # Call step of optimizer to update model params
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        train_loader_tqdm.set_description(f'Ep {epoch} - T {batch_time.val:.2} - T_dat'
                                          f' {data_time.val:.2} L {losses.val:.3}')

        """Log Learning Rate, Benchmarks etc."""
        if i % 10 == 0:
            experiment.log_metric('learning/train_batch10_loss', loss.item(), step=step_batch)

            logger.add_scalar('learning/train_batch10_loss', loss.item(), step_batch)
            logger.add_scalar('data/eval_time', batch_time.val, step_batch)
            logger.add_scalar('data/data_time', data_time.val, step_batch)
            loss_batch10.reset()

        """Log the gradients."""
        if i == 0:
            for name, param in model.named_parameters():
                if 'bn' not in name:
                    logger.add_histogram(name, param, epoch)
                    if param.grad is not None:
                        logger.add_histogram(name + '.grad', param.grad, epoch)

        step_batch += 1

    experiment.log_metric('learning/train_epoch_loss', losses.avg, step=epoch)
    logger.add_scalar('learning/train_epoch_loss', losses.avg, epoch)

    return losses.avg


def test(val_loader, model, criterion, epoch, conf_param, logger, experiment, post_processor, batch_ev, epoch_logger):
    """
    Taken from: https://pytorch.org/tutorials/beginner/aws_distributed_training_tutorial.html
    """
    experiment.set_step(epoch)

    batch_time = eval.MetricMeter()
    losses = eval.MetricMeter()

    inputs = []
    outputs = []
    target_frames = []
    weight_frames = []
    em_outs = []
    tars = []

    """Eval mode."""
    with torch.no_grad():
        end = time.time()
        for i, (x_in, target, weights, em_tar) in enumerate(val_loader):

            x_in = x_in.to(torch.device(conf_param['Hardware']['device']))
            target = target.to(torch.device(conf_param['Hardware']['device']))
            weights = weights.to(torch.device(conf_param['Hardware']['device']))

            # compute output
            output = model(x_in)
            loss_ = criterion(output, target, weights)  # vectorised loss
            loss = loss_.mean()

            # record loss
            losses.update(loss.item())
            criterion.log_batch_loss_cmp(loss_)

            # ToDo: Ugly ...
            """
            Apply non-linearity in the p-channel. Why is this? Because during training non-linearity and loss is
            computationally more efficient if it is combined into one operation. However when we test the model without
            having it to model.eval() no non-linearity in the p channel is applied.
            """
            output = model.apply_pnl(output)

            """Forward output through post-processor for eval."""
            if post_processor is not None:
                em_outs.extend(post_processor.forward(output))

            """Debug here if you want to check (uncomment)"""
            # monitor_io(x_in.detach().cpu(), output.detach().cpu(),
            #            post_processor.forward(output), target.detach().cpu(), em_tar)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            inputs.append(x_in.clone().cpu())
            outputs.append(output.detach().clone().cpu())
            target_frames.append(target.detach().clone().cpu())
            weight_frames.append(weights.detach().clone().cpu())
            tars.extend(deepcopy(em_tar))

            del x_in
            del output
            del target
            del em_tar

    print("Test: Time: {batch_time.avg:.3f} \t""Loss: {loss.avg:.4f}".format(batch_time=batch_time, loss=losses))

    # from list of outputs to one output
    inputs = torch.cat(inputs, 0)
    outputs = torch.cat(outputs, 0)
    target_frames = torch.cat(target_frames, 0)
    weight_frames = torch.cat(weight_frames, 0)

    """Batch evaluation and log"""
    batch_ev.forward(em_outs, tars)
    criterion.log_components(epoch)
    epoch_logger.forward(batch_ev.values, inputs, outputs, target_frames, em_outs, tars, epoch, weight_frames)
    """Plot immediately"""
    # epoch_logger.forward(batch_ev.values, inputs, outputs, target_frames, em_outs, tars, epoch, weight_frames, show=True)

    experiment.log_metric('learning/test_epoch_loss', losses.avg, step=epoch)
    logger.add_scalar('learning/test_epoch_loss', losses.avg, epoch)

    return losses.avg
