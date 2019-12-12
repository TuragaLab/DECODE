from abc import ABC, abstractmethod  # abstract class

import torch
import torch.utils
import tifffile
import csv
from tqdm import tqdm

import deepsmlm.generic.emitter as em
from deepsmlm.generic.utils.data_utils import smlm_collate
from deepsmlm.generic.utils.processing import TransformSequence
from deepsmlm.neuralfitter.dataset import UnsupervisedDataset, SMLMDatasetOnFly, SMLMDatasetOneTimer
from deepsmlm.neuralfitter.pre_processing import ROIOffsetRep, N2C
from deepsmlm.neuralfitter.scale_transform import InverseOffsetRescale, InputFrameRescale


class PredictEval(ABC):
    @abstractmethod
    def __init__(self, model, post_processor, evaluator, batch_size, device='cuda'):
        super().__init__()

        self.model = model
        self.post_processor = post_processor
        self.evaluator = evaluator
        self.batch_size = batch_size
        self.device = torch.device(device)

        self.dataloader = None
        self.gt = None
        self.prediction = None

    def forward(self, output_raw: bool = False):
        """

        :param output_raw: save and output the raw frames
        :return: emitterset (and raw frames if specified).
        """

        # warn the user when he wants to output_raw a big dataset
        if output_raw and self.dataloader.dataset.__len__() > 10000:
            print("WARNING: Are you sure that you want to output the raw frames for this dataset?"
                  " This will mean serious memory consumption.")

        """Eval mode."""
        raw_frames = []
        em_outs = []
        self.model.to(self.device)
        self.model.eval()

        """Eval mode."""
        with torch.no_grad():
            for (x_in, _, _, _) in tqdm(self.dataloader):
                x_in = x_in.to(self.device)

                # compute output
                output = self.model(x_in)
                if output_raw:
                    raw_frames.append(output.detach().cpu())
                """In post processing we need to make sure that we get a single Emitterset for each batch, 
                so that we can easily concatenate."""
                em_outs.append(self.post_processor.forward(output))

        em_merged = em.EmitterSet.cat_emittersets(em_outs, step_frame_ix=self.batch_size)
        self.prediction = em_merged
        if output_raw:
            raw_frames = torch.cat(raw_frames, 0)
            return self.prediction, raw_frames
        else:
            return self.prediction

    def evaluate(self):
        """
        Eval the whole thing. Implement your own method if you need to modify something, e.g. px-size to get proper
        RMSE-vol values. Then call super().evaluate()
        :return:
        """
        if self.evaluator is None:
            print("No Evaluator provided. Cannot perform evaluation.")
            return

        self.evaluator.forward(self.prediction, self.gt)


class PredictEvalSimulation(PredictEval):
    def __init__(self, eval_size, prior, simulator, model, post_processor, evaluator=None, param=None,
                 px_size=100.,
                 device='cuda', batch_size=32, input_preparation=None, multi_frame=True, dataset=None,
                 data_loader=None):
        """

        :param eval_size: how many samples to use for evaluation
        :param prior:
        :param simulator:
        :param model:
        :param post_processor:
        :param evaluator:
        :param param:
        :param px_size:
        :param device:
        :param batch_size:
        :param multi_frame:
        :param dataset:
        :param data_loader:
        """
        super().__init__(model, post_processor, evaluator, batch_size, device)

        self.eval_size = eval_size
        self.prior = prior
        self.simulator = simulator
        self.px_size = px_size
        self.multi_frame = multi_frame
        self.prediction = None
        self.dataset = dataset
        self.dataloader = data_loader
        self.param = param
        self.evaluator = evaluator
        self.input_preparation = input_preparation

        if ((self.param is None) and (self.dataset is None) and (self.dataloader) is None):
            raise ValueError("You need to provide the parameters or you need to provide a dataset and a data loader."
                             "Do the latter if the former fails.")

        if self.input_preparation is None:
            self.input_preparation = TransformSequence([
                N2C(),
                InputFrameRescale.parse(param)
            ])
            print("Setting Input Preparation to: Order Sample axis, Rescale Input Frame.")

        self._init_dataset()

    def _init_dataset(self):
        input_preparation = N2C()

        if self.dataset is None:
            self.dataset = SMLMDatasetOneTimer(None, self.prior, self.simulator, self.eval_size, self.input_preparation,
                                               tar_gen=None, w_gen=None, return_em_tar=True)

        if self.dataloader is None:
            self.dataloader = torch.utils.data.DataLoader(self.dataset,
                                                          batch_size=self.batch_size,
                                                          shuffle=False,
                                                          num_workers=self.param['Hardware']['num_worker_sim'],
                                                          pin_memory=False,
                                                          collate_fn=smlm_collate)

        self.gt = self.dataset.get_gt_emitter('cat')


class PredictEvalTif(PredictEval):
    def __init__(self, tif_stack, activations, model, post_processor, evaluator=None, px_size=100, device='cuda',
                 batch_size=32, multi_frame=True):
        super().__init__(model, post_processor, evaluator, batch_size, device)
        self.tif_stack = tif_stack
        self.activation_file = activations
        self.px_size = px_size
        self.multi_frame = multi_frame

        self.prediction = None
        self.frames = None

    def load_tif(self):
        im = tifffile.imread(self.tif_stack)
        frames = torch.from_numpy(im.astype('float32'))
        frames.unsqueeze_(1)

        self.frames = frames
        self.init_dataset()

    def init_dataset(self, frames=None):
        """
        Initiliase the dataset. Usually by preloaded frames but you can overwrite.
        :param frames: N C(=1) H W
        :return:
        """
        if frames is None:
            frames = self.frames

        ds = UnsupervisedDataset(None, frames=frames,
                                 multi_frame_output=self.multi_frame)
        self.dataloader = torch.utils.data.DataLoader(ds,
                                                      batch_size=self.batch_size, shuffle=False,
                                                      num_workers=8, pin_memory=False)
    def load_csv_(self):
        gt = self.load_csv(self.activation_file)
        self.gt = gt

    @staticmethod
    def load_csv(activation_file):

        if activation_file is None:
            print("WARNING: No activations loaded since file not specified; i.e. there is no ground truth.")
            return

        # read csv
        with open(activation_file) as csv_file:
            csv_reader = csv.reader(csv_file)
            line_count = 0
            id_frame_xyz_camval = []
            for row in csv_reader:
                if line_count == 0:
                    print(row)
                else:
                    id_frame_xyz_camval.append(torch.tensor(
                        (float(row[0]), float(row[1]), float(row[2]), float(row[3]), float(row[4]), float(row[5]))))
                line_count += 1

        id_frame_xyz_camval = torch.stack(id_frame_xyz_camval, 0)

        gt = em.EmitterSet(xyz=id_frame_xyz_camval[:, 2:5], frame_ix=id_frame_xyz_camval[:, 1],
                           phot=id_frame_xyz_camval[:, -1], id=id_frame_xyz_camval[:, 0])
        gt.sort_by_frame()

        return gt


    def load_tif_csv(self):
        self.load_tif()
        self.load_csv()


if __name__ == '__main__':
    from deepsmlm.simulation.emittergenerator import EmitterPopper, EmitterPopperMultiFrame
    from deepsmlm.generic.background import *
    from deepsmlm.generic.emitter import *
    from deepsmlm.generic.plotting.frame_coord import *
    from deepsmlm.generic.inout.load_calibration import *
    from deepsmlm.generic.psf_kernel import *
    from deepsmlm.generic.noise import *
    from deepsmlm.generic.utils.data_utils import *
    from deepsmlm.simulation.simulator import *
    from deepsmlm.simulation.structure_prior import *
    from deepsmlm.neuralfitter.dataset import *
    from deepsmlm.neuralfitter.models.model import *
    from deepsmlm.neuralfitter.models.model_offset import *
    from deepsmlm.neuralfitter.models.model_beta import *
    from deepsmlm.neuralfitter.models.model_param import *
    from deepsmlm.neuralfitter.pre_processing import *
    from deepsmlm.neuralfitter.post_processing import *
    from deepsmlm.evaluation.match_emittersets import *
    from deepsmlm.neuralfitter.scale_transform import *
    from deepsmlm.generic.inout.load_save_model import *
    from deepsmlm.neuralfitter.pred_tif import *
    from deepsmlm.evaluation.evaluation import *
    import deepsmlm.generic.utils.processing as processing
    from deepsmlm.neuralfitter.arguments import *
    from deepsmlm.generic.phot_camera import *
    import deepsmlm.generic.inout.write_load_param as wlp

    deepsmlm_root = '/home/lucas/RemoteDeploymentTemp/DeepSMLMv2/'
    os.chdir(deepsmlm_root)

    calibration_file = deepsmlm_root + 'data/Calibration/SMLM Challenge Beads/Coefficients Big ROI/AS-Exp_100nm_3dcal.mat'

    model_file = 'network_central/2019-12-08/simple_net_3steps_40em_gnskip0_strongwd_7.pt'
    param_file = 'network_central/2019-12-08/simple_net_3steps_40em_gnskip0_strongwd_param.json'

    param = wlp.ParamHandling().load_params(param_file)
    model = SimpleSMLMNet.parse(param)
    model = LoadSaveModel(model, None, input_file=model_file).load_init()

    post_processor = processing.TransformSequence.parse([OffsetRescale,
                                                         Offset2Coordinate,
                                                         ConsistencyPostprocessing], param)

    # setup evaluation

    matcher = GreedyHungarianMatching(dist_lat=150.)
    segmentation_eval = SegmentationEvaluation(False)
    distance_eval = DistanceEvaluation(False)

    batch_size = 64
    evaluation = BatchEvaluation(matcher, segmentation_eval, distance_eval, px_size=torch.tensor([100., 100., 1.]),
                                 batch_size=batch_size)

    # specify how big the evaluation set should be
    ds_size = 256

    smap_psf = SMAPSplineCoefficient(calibration_file)
    psf = smap_psf.init_spline(param['Simulation']['psf_extent'][0],
                               param['Simulation']['psf_extent'][1],
                               param['Simulation']['img_size'])

    structure_prior = RandomStructure(param['Simulation']['emitter_extent'][0],
                                      param['Simulation']['emitter_extent'][1],
                                      param['Simulation']['emitter_extent'][2])
    frame_range = (-1, 1)
    prior = EmitterPopperMultiFrame(structure_prior,
                                    density=param['Simulation']['density'],
                                    intensity_mu_sig=param['Simulation']['intensity_mu_sig'],
                                    lifetime=param['Simulation']['lifetime_avg'],
                                    num_frames=3,
                                    emitter_av=2)

    bg = processing.TransformSequence.parse([UniformBackground,
                                             PerlinBackground], param)

    noise = Photon2Camera.parse(param)
    simulator = Simulation(None,
                           param['Simulation']['emitter_extent'],
                           psf,
                           background=bg,
                           noise=noise,
                           poolsize=0,
                           frame_range=frame_range)

    pred = PredictEvalSimulation(ds_size, prior, simulator, model, post_processor, evaluation, param=param,
                                 px_size=100.)

    _, raw = pred.forward(True)
    print("Done.")


