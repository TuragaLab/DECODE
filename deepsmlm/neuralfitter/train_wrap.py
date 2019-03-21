import datetime
import os
import sys

import torch
from tensorboardX import SummaryWriter
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from deepsmlm.generic.inout.load_calibration import SMAPSplineCoefficient
from deepsmlm.generic.inout.load_save_emitter import NumpyInterface
from deepsmlm.generic.inout.load_save_model import LoadSaveModel
from deepsmlm.generic.noise import Poisson
from deepsmlm.generic.psf_kernel import ListPseudoPSFInSize
from deepsmlm.neuralfitter.arguments import Args
from deepsmlm.neuralfitter.dataset import SMLMDataset
from deepsmlm.neuralfitter.dataset import SMLMDatasetOnFly
from deepsmlm.neuralfitter.losscollection import MultiScaleLaplaceLoss
from deepsmlm.neuralfitter.model import DenseLoco
from deepsmlm.neuralfitter.pre_processing import N2C, SingleEmitterOnlyZ
from deepsmlm.neuralfitter.train_test import train, test
from deepsmlm.simulator.emittergenerator import EmitterPopper
from deepsmlm.simulator.simulator import Simulation

"""Several pseudo-global variables useful for data processing and debugging."""
deepsmlm_root = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)),
                 os.pardir, os.pardir)) + '/'


if __name__ == '__main__':

    """TensorbaordX Log"""

    log_dir = deepsmlm_root + 'log/' + str(datetime.datetime.now())[:16]
    logger = SummaryWriter(log_dir)

    if len(sys.argv) == 1:  # no .ini file specified
        dataset_file = deepsmlm_root + \
                       'data/2019-03-19/complete_z_range.npz'
        weight_out = deepsmlm_root + 'network/2019-03-21/model_3_fordemo.pt'
        weight_in = None  # deepsmlm_root + 'network/2019-03-21/model_1.pt'

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

        data_smlm = SMLMDataset(emitter, extent, frames, target_generator, multi_frame_output=False,
                                dimensionality=None)

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

        train_loader = DataLoader(train_data_smlm, batch_size=128, shuffle=False, num_workers=12, pin_memory=True)
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

    """Log the model"""
    dummy = torch.rand((32, 1, img_shape[0], img_shape[1]), requires_grad=True)
    logger.add_graph(model, dummy, False)

    model_ls = LoadSaveModel(model,
                             weight_out,
                             cuda=args.cuda,
                             input_file=weight_in)
    model = model_ls.load_init()
    logger.add_text('args/model_input', 'None' if (weight_in is None) else weight_in)
    logger.add_text('args/model_output', 'None' if (weight_out is None) else weight_out)
    logger.add_text('args/device', 'CUDA' if (args.cuda is True) else 'CPU')

    if args.cuda:
        model = model.cuda()

    optimiser = Adam(model.parameters(), lr=1E-4)
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
                                  patience=20,
                                  threshold=0.0001,
                                  cooldown=10,
                                  verbose=True)

    # milestones = [300, 500, 600, 700, 800, 900, 1000]
    # scheduler = MultiStepLR(optimiser, milestones=milestones, gamma=0.5)

    """Ask if everything is correct before we start."""
    args.print_confirmation()
    for i in range(args.epochs):
        # log the learning rate
        logger.add_scalar('learning/learning_rate', optimiser.param_groups[0]['lr'], i)

        train(train_loader, model, optimiser, criterion, i, args, logger)
        val_loss = test(test_loader, model, criterion, i, args, logger)
        scheduler.step(val_loss)

        if i % 1 == 0:
            model_ls.save(model)
