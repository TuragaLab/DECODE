import os
import pprint
import sys
import time
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from deepsmlm.neuralfitter.dataset import SMLMDataset
from deepsmlm.generic.io.load_save_model import LoadSaveModel
from deepsmlm.generic.io.load_save_emitter import MatlabInterface
from deepsmlm.neuralfitter.losscollection import BumpMSELoss, BumpMSELoss3DzLocal


class Args:
    """
    Convenience for training arguments.
    """
    def __init__(self, cuda=True, epochs=100, num_prints=5, sm_sigma=1,
                 root_folder=None, data_path=None, model_in_path=None, model_out_path=None):
        self.cuda = cuda if torch.cuda.is_available() else False
        self.epochs = epochs
        self.num_prints = num_prints
        self.sm_sigma = sm_sigma

        self.root_folder = root_folder
        self.data_path = data_path
        self.model_in_path = model_in_path
        self.model_out_path = model_out_path

    def print_confirmation(self):
        """
        Print arguments and wait for confirmation.
        :return: void
        """
        print('The configured arguments are:')
        pp = pprint.PrettyPrinter(width=-1)
        pp.pprint(vars(self))
        input('Press Enter to continue ...')


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def train(train_loader, model, optimizer, criterion, epoch):

    print('Epoch: [{}] \t lr: {}'.format(epoch, optimizer.defaults['lr']))

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    print_steps = torch.round(torch.linspace(0, train_loader.__len__(), args.num_prints))

    model.train()
    end = time.time()
    for i, (input, target, _) in enumerate(train_loader):

        # measure data loading time
        data_time.update(time.time() - end)

        """Plot here to check whether what you feed the network is what we would expect."""
        # if True:
        #     import matplotlib.pyplot as plt
        #     plt.subplot(121)
        #     plt.imshow(input[0, 1, :, :])
        #     plt.subplot(122)
        #     plt.imshow(target[0, 0, :, :].detach().numpy())
        #     plt.show()

        if args.cuda:  # model_deep.cuda():
            input, target = input.cuda(), target.cuda()

        # compute output
        output = model(input)
        loss = criterion(output, target)

        # measure accuracy and record loss
        # prec1, prec5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        # top1.update(prec1[0], input.size(0))
        # top5.update(prec5[0], input.size(0))

        # compute gradients in a backward pass
        optimizer.zero_grad()
        loss.backward()

        # Call step of optimizer to update model params
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i in print_steps:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses))
                  # 'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  # 'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'


def test(val_loader, model, criterion):
    """
    Taken from: https://pytorch.org/tutorials/beginner/aws_distributed_training_tutorial.html
    """

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    print_steps = torch.round(torch.linspace(0, val_loader.__len__(), 3))

    # switch to evaluate mode
    model.eval()
    with torch.no_grad():
        end = time.time()
        for i, (input, target, _) in enumerate(val_loader):

            if args.cuda:
                input = input.cuda(non_blocking=True)
                target = target.cuda(non_blocking=True)

            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            # prec1, prec5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            # top1.update(prec1[0], input.size(0))
            # top5.update(prec5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i in print_steps:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                       i, len(val_loader), batch_time=batch_time, loss=losses))
                      # 'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      # 'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.

        # print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
        #       .format(top1=top1, top5=top5))
    return losses.avg


if __name__ == '__main__':

    deepsmlm_root = os.path.abspath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     os.pardir, os.pardir)) + '/'

    if len(sys.argv) == 1:  # no .ini file specified
        dataset_file = deepsmlm_root + 'data/2019-02-20 Pores/simulation.mat'
        weight_out = deepsmlm_root + 'network/2019-02-20 Pores/camera_units.pt'
        weight_in = None # deepsmlm_root + 'network/2019-02-19 Spline Z Without Z Prediction/spline_1e4_easy_z_10bg_20190215.pt'

    else:
        dataset_file = deepsmlm_root + sys.argv[1]
        weight_out = deepsmlm_root + sys.argv[2]
        weight_in = None if sys.argv[3].__len__() == 0 else deepsmlm_root + sys.argv[3]

    args = Args(cuda=True,
                epochs=100,
                num_prints=5,
                sm_sigma=1,
                root_folder=deepsmlm_root,
                data_path=dataset_file,
                model_out_path=weight_out,
                model_in_path=weight_in)

    """Load Data from binary."""
    emitter, extent, frames = MatlabInterface().load_binary(dataset_file)
    data_smlm = SMLMDataset(emitter, extent, frames)

    """The model load and save interface."""
    model_ls = LoadSaveModel(weight_out,
                             cuda=args.cuda,
                             input_file=weight_in)
    model = model_ls.load_init()

    if args.cuda:  # move model to CUDA device
        model = model.cuda()

    optimiser = Adam(model.parameters(), lr=0.0001)

    """Get loss function."""
    criterion = BumpMSELoss(kernel_sigma=args.sm_sigma, cuda=args.cuda, l1_f=0.1).return_criterion()
    # criterion = BumpMSELoss3DzLocal(kernel_sigma_photons=args.sm_sigma,
    #                                 kernel_sigma_z=5,
    #                                 cuda=args.cuda,
    #                                 phot_thres=100, l1_f=0.1, d3_f=1).return_criterion()

    """Learning Rate Scheduling"""
    scheduler = ReduceLROnPlateau(optimiser, mode='min', patience=4, threshold=0.02)

    train_size = int(0.9 * len(data_smlm))
    test_size = len(data_smlm) - train_size
    train_data, test_data = torch.utils.data.random_split(data_smlm, [train_size, test_size])

    train_loader = DataLoader(train_data, batch_size=32, shuffle=True, num_workers=12, pin_memory=True)
    test_loader = DataLoader(test_data, batch_size=32, shuffle=False, num_workers=2, pin_memory=True)

    """Ask if everything is correct before we start."""
    args.print_confirmation()
    for i in range(args.epochs):

        train(train_loader, model, optimiser, criterion, i)
        val_loss = test(test_loader, model, criterion)
        scheduler.step(val_loss)

        model_ls.save(model)

