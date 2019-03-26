import os
import random
import time

import numpy as np
import torch
from matplotlib import pyplot as plt

from deepsmlm.generic.plotting.frame_coord import PlotFrameCoord, PlotCoordinates3D
from deepsmlm.evaluation.evaluation import AverageMeter


"""Several pseudo-global variables useful for data processing and debugging."""
deepsmlm_root = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)),
                 os.pardir, os.pardir)) + '/'

DEBUG = True
LOG = True
TENSORBOARD = True

LOG = True if DEBUG else LOG


def plot_io_model(input, output, target, ix):
    pass


def train(train_loader, model, optimizer, criterion, epoch, hy_par, logger, experiment):
    last_print_time = 0
    loss_values = []
    step_batch = epoch * train_loader.__len__()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    model.train()
    end = time.time()
    for i, (input, target, _) in enumerate(train_loader):

        if (epoch == 0) and (i == 0) and LOG:
            """Save a batch to see what we input into the network."""
            debug_file = deepsmlm_root + 'data/debug.pt'
            torch.save((input, target), debug_file)
            print("LOG: I saved a batch for you. Look what the network sees for verification purpose.")

        # measure data loading time
        data_time.update(time.time() - end)

        input = input.to(hy_par.device)
        if type(target) is torch.Tensor:
            target = target.to(hy_par.device)
        elif type(target) in (tuple, list):
            target = (target[0].to(hy_par.device), target[1].to(hy_par.device))
        else:
            raise TypeError("Not supported type to push to cuda.")

        # compute output
        output = model(input)
        loss = criterion(output, target)

        # record loss
        losses.update(loss.item(), input.size(0))
        loss_values.append(loss.item())

        # compute gradients in a backward pass
        optimizer.zero_grad()
        loss.backward()

        # Call step of optimizer to update model params
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if (i in [0, 1, 2, 10]) or (time.time() > last_print_time + 1):  # print the first few batches plus after 1s
            last_print_time = time.time()
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses))

        """Log Learning Rate, Benchmarks etc."""
        if i % 10 == 0:
            experiment.log_metric('learning/train_10_batch_loss', np.mean(loss_values), step=step_batch)

            logger.add_scalar('learning/train_loss', np.mean(loss_values), step_batch)
            logger.add_scalar('data/eval_time', batch_time.val, step_batch)
            logger.add_scalar('data/data_time', data_time.val, step_batch)
            loss_values.clear()

        if (epoch == 0) and (i == 0):
            num_plot = 5
            figures = []
            for p in range(num_plot):
                ix_in_batch = random.randint(0, input.shape[0] - 1)
                channel = 0 if (input.shape[1] == 1) else 1
                img = input[ix_in_batch, channel, :, :].detach().cpu()
                xyz_tar = target[0][ix_in_batch, :].detach().cpu()
                xyz_out = (output[0][ix_in_batch, :]).detach().cpu()
                phot_tar = target[1][ix_in_batch].detach().cpu()
                fig = plt.figure()
                PlotFrameCoord(frame=img, pos_tar=xyz_tar, phot_tar=phot_tar).plot()
                plt.title('Sample in Batch: {} - Channel: {}'.format(ix_in_batch, channel))
                figures.append(fig)
                fig_str = 'training/fig_{}'.format(p)
                plt.show()
                experiment.log_figure(fig_str, fig)
                logger.add_figure(fig_str, fig, epoch)

        if i == 0:
            for name, param in model.named_parameters():
                if 'bn' not in name:
                    logger.add_histogram(name, param, epoch)
                    logger.add_histogram(name + '.grad', param.grad, epoch)

        step_batch += 1


def test(val_loader, model, criterion, epoch, hy_par, logger, experiment):
    """
    Taken from: https://pytorch.org/tutorials/beginner/aws_distributed_training_tutorial.html
    """
    experiment.set_step(epoch)

    batch_time = AverageMeter()
    losses = AverageMeter()

    # switch to evaluate mode
    model.eval()
    with torch.no_grad():
        end = time.time()
        for i, (input, target, _) in enumerate(val_loader):

            input = input.to(hy_par.device)
            if type(target) is torch.Tensor:
                target = target.to(hy_par.device)
            elif type(target) in (tuple, list):
                target = (target[0].to(hy_par.device), target[1].to(hy_par.device))
            else:
                raise TypeError("Not supported type to push to cuda.")

            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            losses.update(loss.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if (i in [0, 1, 2, 10]) or (i % 200 == 0):
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                    i, len(val_loader), batch_time=batch_time, loss=losses))

            """Log first 3 and a random subset to tensorboard"""
            if i == 0:
                num_plot = 5
                for p in range(num_plot):
                    ix_in_batch = p  # random.randint(0, input.shape[0] - 1)
                    channel = 0 if (input.shape[1] == 1) else 1
                    img = input[ix_in_batch, channel, :, :].cpu()
                    xyz_tar = target[0][ix_in_batch, :].cpu()
                    xyz_out = (output[0][ix_in_batch, :]).cpu()
                    phot_tar = target[1][ix_in_batch].cpu()
                    phot_out = output[1][ix_in_batch].cpu()
                    fig = plt.figure()
                    PlotFrameCoord(frame=img, pos_tar=xyz_tar, pos_out=xyz_out, phot_out=phot_out).plot()
                    plt.title('Sample in Batch: {} - Channel: {}'.format(ix_in_batch, channel))
                    fig_str = 'testset/fig_{}'.format(p)
                    logger.add_figure(fig_str, fig, epoch)

                    fig = plt.figure()
                    PlotCoordinates3D(pos_tar=xyz_tar, pos_out=xyz_out, phot_out=phot_out).plot()
                    fig_str = 'testset/fig_{}_3d'.format(p)
                    experiment.log_figure(fig_str, fig)
                    logger.add_figure(fig_str, fig, epoch)

    experiment.log_metric('learning/test_loss', losses.val, epoch)
    logger.add_scalar('learning/test_loss', losses.val, epoch)
    return losses.val
