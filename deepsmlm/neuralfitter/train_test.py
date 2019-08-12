import os
import random
import time

import numpy as np
import torch
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


def train(train_loader, model, optimizer, criterion, epoch, conf_param, logger, experiment, calc_new):
    last_print_time = 0
    loss_values = []
    step_batch = epoch * train_loader.__len__()

    batch_time = eval.MetricMeter()
    data_time = eval.MetricMeter()
    losses = eval.MetricMeter()
    loss_batch10 = eval.MetricMeter()

    model.train()
    end = time.time()
    for i, (x_in, target) in enumerate(train_loader):

        if (epoch == 0) and (i == 0) and LOG:
            """Save a batch to see what we input into the network."""
            debug_file = deepsmlm_root + 'data/debug.pt'
            torch.save((x_in, target), debug_file)
            print("LOG: I saved a batch for you. Look what the network sees for verification purpose.")

        # measure data loading time
        data_time.update(time.time() - end)

        x_in = x_in.to(torch.device(conf_param['Hyper']['device']))
        if type(target) is torch.Tensor:
            target = target.to(torch.device(conf_param['Hyper']['device']))
        elif type(target) in (tuple, list):
            target = (target[0].to(torch.device(conf_param['Hyper']['device'])), target[1].to(torch.device(param['Hyper']['device'])))
        else:
            raise TypeError("Not supported type to push to cuda.")

        # compute output
        output = model(x_in)

        """Ignore the loss of the boundary frames"""
        if conf_param['Hyper']['ignore_boundary_frames']:
            loss_ = criterion(output[1:-1], target[1:-1])
        else:
            loss_ = criterion(output, target)

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

        """Print 0th, 1st, 2nd, 10th and every 5 secs to the console."""
        if (i in [0, 1, 2, 10]) or (time.time() > last_print_time + 5):  # print the first few batches plus after 5s
            last_print_time = time.time()
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses))

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
    em_outs = []
    tars = []

    """Eval mode."""
    with torch.no_grad():
        end = time.time()
        for i, (x_in, target, em_tar) in enumerate(val_loader):

            x_in = x_in.to(torch.device(conf_param['Hyper']['device']))
            if type(target) is torch.Tensor:
                target = target.to(torch.device(conf_param['Hyper']['device']))
            elif type(target) in (tuple, list):
                target = (target[0].to(torch.device(conf_param['Hyper']['device'])),
                          target[1].to(torch.device(conf_param['Hyper']['device'])))
            else:
                raise TypeError("Not supported type to push to cuda.")

            # compute output
            output = model(x_in)
            loss_ = criterion(output, target)  # vectorised loss
            loss = loss_.mean()

            # record loss
            losses.update(loss.item())
            criterion.log_batch_loss_cmp(loss_)

            # ToDo: Ugly ...
            # apply non-linearity in the p-channel
            output = model.apply_pnl(output)

            """Forward output through post-processor for eval."""
            if post_processor is not None:
                em_outs.extend(post_processor.forward(output))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            inputs.append(x_in.cpu())
            outputs.append(output.detach().cpu())
            target_frames.append(target.detach().cpu())
            tars.extend(em_tar)

    print("Test: Time: {batch_time.avg:.3f} \t""Loss: {loss.avg:.4f}".format(batch_time=batch_time, loss=losses))

    # from list of outputs to one output
    inputs = torch.cat(inputs, 0)
    outputs = torch.cat(outputs, 0)
    target_frames = torch.cat(target_frames, 0)

    """Batch evaluation and log"""
    batch_ev.forward(em_outs, tars)
    criterion.log_components(epoch)
    epoch_logger.forward(batch_ev.values, inputs, outputs, target_frames, em_outs, tars, epoch)

    experiment.log_metric('learning/test_epoch_loss', losses.avg, step=epoch)
    logger.add_scalar('learning/test_epoch_loss', losses.avg, epoch)

    return losses.avg
