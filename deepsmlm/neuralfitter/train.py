import sys
import time
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader

from deepsmlm.neuralfitter.dataset import SMLMDataset
from deepsmlm.generic.io.load_save_model import LoadSaveModel
from deepsmlm.generic.io.load_save_emitter import MatlabInterface
from deepsmlm.generic.noise import GaussianSmoothing
from deepsmlm.neuralfitter.losscollection import bump_mse_loss


class Args:
    """
    Convenience for training options.
    """
    def __init__(self, cuda=True, epochs=100, num_prints=5, sm_sigma=1):
        self.cuda = cuda if torch.cuda.is_available() else False
        self.epochs = epochs
        self.num_prints = num_prints
        self.sm_sigma = sm_sigma


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

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    print_steps = torch.round(torch.linspace(0, train_loader.__len__(), args.num_prints))

    model.train()
    end = time.time()
    for i, (input, target) in enumerate(train_loader):

        if args.cuda:  # model_deep.cuda():
            input, ground_truth = input.cuda(), ground_truth.cuda()

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

    print_steps = torch.round(torch.linspace(0, val_loader.__len__(), args.num_prints))

    # switch to evaluate mode
    model.eval()
    end = time.time()
    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):

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


if __name__ == '__main__':
    if len(sys.argv) == 1:  # no .ini file specified
        dataset_file = '/Users/lucasmueller/Repositories/deepsmlm/data/spline_1e3_noz.mat'
        weight_out = '/Users/lucasmueller/Repositories/deepsmlm/network/spline_1e3_noz.pt'
        weight_in = None  # '../../network/spline_1e4_no_z.pt'

    else:
        dataset_file = sys.argv[1]
        weight_out = sys.argv[2]
        weight_in = None if sys.argv[3].__len__() == 0 else sys.argv[3]

    args = Args(cuda=True,
                epochs=1000,
                num_prints=5,
                sm_sigma=1)

    data_smlm = SMLMDataset(MatlabInterface().load_binary, dataset_file)
    model_ls = LoadSaveModel(weight_out,
                             cuda=args.cuda,
                             warmstart_file=weight_in)

    model = model_ls.load_init()

    optimiser = Adam(model.parameters(), lr=0.001)

    gaussian_kernel = GaussianSmoothing(1, [7, 7], args.sm_sigma, dim=2, cuda=args.cuda,
                                        padding=lambda x: F.pad(x, [3, 3, 3, 3], mode='reflect'))

    def criterion(input, target):
        return bump_mse_loss(input, target,
                             kernel_pred=gaussian_kernel,
                             kernel_true=gaussian_kernel)

    if args.cuda:
        model = model.cuda()

    train_size = int(0.9 * len(data_smlm))
    test_size = len(data_smlm) - train_size
    train_data, test_data = torch.utils.data.random_split(data_smlm, [train_size, test_size])

    train_loader = DataLoader(train_data, batch_size=1, shuffle=True, num_workers=12)
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=4)

    for i in range(args.epochs):
        train(train_loader, model, optimiser, criterion, i)
        test(test_loader, model, criterion)
        model_ls.save(model)
