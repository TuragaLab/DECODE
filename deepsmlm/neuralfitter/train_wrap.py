import datetime
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import sys
import time
import torch
from tensorboardX import SummaryWriter
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, MultiStepLR
from torch.utils.data import DataLoader

from deepsmlm.generic.plotting.live_plotting import Visualizations
from deepsmlm.neuralfitter.arguments import Args
from deepsmlm.evaluation.evaluation import AverageMeter
from deepsmlm.neuralfitter.dataset import SMLMDataset
from deepsmlm.neuralfitter.model import DeepSMLN, DeepLoco, DenseLoco
from deepsmlm.neuralfitter.model_beta import EncoderFC, SuperDumbFCNet, DenseNetResNet
from deepsmlm.neuralfitter.model_densenet import DenseNet
from deepsmlm.generic.inout.load_save_model import LoadSaveModel
from deepsmlm.generic.inout.load_save_emitter import MatlabInterface, NumpyInterface
from deepsmlm.neuralfitter.losscollection import BumpMSELoss, BumpMSELoss3DzLocal, MultiScaleLaplaceLoss
from deepsmlm.generic.inout.load_calibration import SMAPSplineCoefficient, StormAnaCoefficient
from deepsmlm.generic.psf_kernel import DeltaPSF, DualDelta, ListPseudoPSF, SplineCPP, ListPseudoPSFInSize
from deepsmlm.neuralfitter.pre_processing import RemoveOutOfFOV, N2C, Identity, SingleEmitterOnlyZ, ZasClassification, ZasSimpleRegression
from deepsmlm.generic.noise import IdentityNoise, Poisson
from deepsmlm.simulator.simulator import Simulation
from deepsmlm.simulator.emittergenerator import EmitterPopper, EmitterPopperMultiFrame
from deepsmlm.neuralfitter.dataset import SMLMDatasetOnFly

"""Several pseudo-global variables useful for data processing and debugging."""
deepsmlm_root = os.path.abspath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     os.pardir, os.pardir)) + '/'

DEBUG = True
LOG = True
TENSORBOARD = True

LOG = True if DEBUG else LOG


def train(train_loader, model, optimizer, criterion, epoch):

    # print('Epoch: [{}] \t lr: {}'.format(epoch, optimizer.defaults['lr']))
    last_print_time = 0
    loss_values = []
    global step_batch

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    print_steps = torch.round(torch.linspace(0, train_loader.__len__(), args.num_prints))

    model.train()
    end = time.time()
    for i, (input, target, _) in enumerate(train_loader):

        if LOG:
            if (epoch == 0) and (i == 0):
                """Save a batch to see what we input into the network."""
                debug_file = deepsmlm_root + 'data/debug.pt'
                torch.save((input, target), debug_file)
                print("LOG: I saved a batch for you. Look what the network sees for verification purpose.")

        if DEBUG:
            if (epoch == 0) and (i == 0):
                from deepsmlm.generic.plotting.frame_coord import PlotFrameCoord
                num_plot = 3

                for p in range(num_plot):
                    ix_in_batch = random.randint(0, input.shape[0] - 1)
                    for channel in range(input.shape[1]):

                        img = input[ix_in_batch, channel, :, :]
                        xyz_tar = target[0][ix_in_batch, :]
                        phot_tar = target[1][ix_in_batch]
                        PlotFrameCoord(frame=img, pos_tar=xyz_tar).plot()
                        plt.title('Sample in Batch: {} - Channel: {}'.format(ix_in_batch, channel))
                        plt.show()

        # measure data loading time
        data_time.update(time.time() - end)

        if args.cuda:  # model_deep.cuda():
            input = input.cuda()
            if type(target) is torch.Tensor:
                target = target.cuda()
            elif type(target) in (tuple, list):
                target = (target[0].cuda(), target[1].cuda())
            else:
                raise TypeError("Not supported type to push to cuda.")

        # compute output
        output = model(input)
        loss = criterion(output, target)

        # measure accuracy and record loss
        # prec1, prec5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        loss_values.append(loss.item())
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

        if (i in [0, 1, 2, 10]) or (time.time() > last_print_time + 1):
            last_print_time = time.time()
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses))
                  # 'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  # 'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'

        if i % 10 == 0:
            writer.add_scalar('data/train_loss', np.mean(loss_values), step_batch)
            writer.add_scalar('data/eval_time', batch_time.val, step_batch)
            writer.add_scalar('data/data_time', data_time.val, step_batch)
            loss_values.clear()
        step_batch += 1


def test(val_loader, model, criterion, epoch):
    """
    Taken from: https://pytorch.org/tutorials/beginner/aws_distributed_training_tutorial.html
    """

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # print_steps = torch.round(torch.linspace(0, val_loader.__len__(), 3))

    # switch to evaluate mode
    model.eval()
    with torch.no_grad():
        end = time.time()
        for i, (input, target, _) in enumerate(val_loader):

            if args.cuda:  # model_deep.cuda():
                input = input.cuda()
                if type(target) is torch.Tensor:
                    target = target.cuda()
                elif type(target) in (tuple, list):
                    target = (target[0].cuda(), target[1].cuda())
                else:
                    raise TypeError("Not supported type to push to cuda.")

            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            # prec1, prec5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if (i in [0, 1, 2, 10]) or (i % 200 == 0):
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                       i, len(val_loader), batch_time=batch_time, loss=losses))
                      # 'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      # 'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.

    writer.add_scalar('data/test_loss', losses.val, epoch)
    return losses.val


if __name__ == '__main__':

    """TensorbaordX Log"""

    log_dir = deepsmlm_root + 'log/' + str(datetime.datetime.now())[:16]
    writer = SummaryWriter(log_dir)

    step_batch = 0

    if len(sys.argv) == 1:  # no .ini file specified
        dataset_file = deepsmlm_root + \
                       'data/2019-03-19/complete_z_range.npz'
        weight_out = deepsmlm_root + 'network/2019-03-20 DenseNetResNet/model_denseloco_1.pt'
        weight_in = deepsmlm_root + 'network/2019-03-20 DenseNetResNet/model_denseloco.pt'

    else:
        dataset_file = deepsmlm_root + sys.argv[1]
        weight_out = deepsmlm_root + sys.argv[2]
        weight_in = None if sys.argv[3].__len__() == 0 else deepsmlm_root + sys.argv[3]

    args = Args(cuda=True,
                epochs=10000,
                num_prints=5,
                sm_sigma=1,
                root_folder=deepsmlm_root,
                data_path=dataset_file,
                model_out_path=weight_out,
                model_in_path=weight_in)

    mode = 'Online'

    if mode == 'PreComputedSamples':
        """Load Data from binary."""
        emitter, extent, frames = NumpyInterface().load_binary(dataset_file)
        # emitter, extent, frames = MatlabInterface().load_binary(dataset_file)

        # target_generator = ZasSimpleRegression()
        target_generator = SingleEmitterOnlyZ()
        # target_generator = ListPseudoPSF(zero_fill_to_size=64,
        #                                  dim=3)

        data_smlm = SMLMDataset(emitter, extent, frames, target_generator, multi_frame_output=False, dimensionality=None)

        split_ratio = 0.9
        test_size = 1024
        train_size = data_smlm.__len__() - test_size
        train_dataset, test_dataset = torch.utils.data.random_split(data_smlm, [train_size, test_size])

        train_data_smlm, test_data_smlm = torch.utils.data.random_split(data_smlm, [train_size, test_size])

        train_loader = DataLoader(train_data_smlm, batch_size=32, shuffle=True, num_workers=24, pin_memory=True)
        test_loader = DataLoader(test_data_smlm, batch_size=1024, shuffle=False, num_workers=12, pin_memory=True)

    elif mode == 'Online':
        """Load 'Dataset' which is generated on the fly."""
        sim_extent = ((-0.5, 31.5), (-0.5, 31.5), (-500, 500))
        psf_extent = ((-0.5, 31.5), (-0.5, 31.5), (-750., 750.))
        img_shape = (32, 32)
        spline_file = deepsmlm_root + \
                      'data/Cubic Spline Coefficients/2019-02-20/60xOil_sampleHolderInv__CC0.140_1_MMStack.ome_3dcal.mat'
        psf = SMAPSplineCoefficient(spline_file).init_spline(psf_extent[0], psf_extent[1], img_shape)
        noise = Poisson(bg_uniform=10)

        # psf = GaussianExpect(xextent=extent[0], yextent=extent[1], zextent=None, img_shape=img_shape,
        #                      sigma_0=(1.5, 1.5))
        prior = EmitterPopper(sim_extent[0], sim_extent[1], sim_extent[2], density=0.003, photon_range=(4000, 10000))
        # prior = EmitterPopperMultiFrame(sim_extent[0], sim_extent[1], sim_extent[2],
        #                                 density=0.005,
        #                                 photon_range=(1000, 4000),
        #                                 lifetime=1,
        #                                 num_frames=3)
        simulator = Simulation(None,
                               sim_extent,
                               psf,
                               noise,
                               poolsize=0,
                               frame_range=(-1, -1))

        input_preparation = N2C()

        target_generator = ListPseudoPSFInSize(sim_extent[0], sim_extent[1], sim_extent[2], zts=64, dim=3)

        train_data_smlm = SMLMDatasetOnFly(None, prior, simulator, (256 * 10 * 10),
                                           input_preparation, target_generator, None, reuse=False)

        test_data_smlm = SMLMDatasetOnFly(None, prior, simulator, 256,
                                          input_preparation, target_generator, None, reuse=True)

        train_loader = DataLoader(train_data_smlm, batch_size=256, shuffle=False, num_workers=12, pin_memory=True)
        test_loader = DataLoader(test_data_smlm, batch_size=512, shuffle=False, num_workers=4, pin_memory=True)

    else:
        raise NameError("You used the wrong switch of how to get the training data.")

    """Model load and save interface."""
    model = DenseLoco(extent=psf_extent, ch_in=1, dim_out=3)
    # model = DenseNetResNet()
    # model = DenseNet(num_classes=1, num_channels=1)
    # model = SuperDumbFCNet(676, None)
    # model = EncoderFC(limits=(extent[2][0], extent[2][1]))
    # model = DeepLoco(extent=sim_extent,
    #                  ch_in=3,
    #                  dim_out=3)

    #dummy = model(torch.rand((1, 1, 32, 32), requires_grad=True))
    #writer.add_graph(model, dummy)

    model_ls = LoadSaveModel(model,
                             weight_out,
                             cuda=args.cuda,
                             input_file=weight_in)
    model = model_ls.load_init()

    if args.cuda:
        model = model.cuda()

    optimiser = Adam(model.parameters(), lr=5E-5)
    # optimiser = SGD(model.parameters(), lr=0.000001)

    """Loss function."""
    # criterion = torch.nn.NLLLoss()
    # criterion = torch.nn.MSELoss()
    criterion = MultiScaleLaplaceLoss(kernel_sigmas=(0.64, 3.20, 6.4, 19.2)).return_criterion()
    # criterion = BumpMSELoss(kernel_sigma=args.sm_sigma, cuda=args.cuda, l1_f=0.1).return_criterion()

    """Learning Rate Scheduling"""
    scheduler = ReduceLROnPlateau(optimiser,
                                  mode='min',
                                  factor=0.5,
                                  patience=10,
                                  threshold=0.05,
                                  cooldown=10,
                                  verbose=True)

    # milestones = [300, 500, 600, 700, 800, 900, 1000]
    # scheduler = MultiStepLR(optimiser, milestones=milestones, gamma=0.5)

    """Visualisation"""
    # vis = Visualizations()

    """Ask if everything is correct before we start."""
    args.print_confirmation()
    for i in range(args.epochs):
        writer.add_scalar('data/learning_rate', optimiser.param_groups[0]['lr'], i)
        train(train_loader, model, optimiser, criterion, i)
        val_loss = test(test_loader, model, criterion, i)
        scheduler.step(val_loss)
        if i % 1 == 0:
            model_ls.save(model)
