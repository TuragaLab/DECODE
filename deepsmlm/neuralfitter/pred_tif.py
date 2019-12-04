from abc import ABC, abstractmethod  # abstract class

import torch
import torch.utils
import tifffile
import csv
from tqdm import tqdm

import deepsmlm.generic.emitter as em
from deepsmlm.generic.utils.data_utils import smlm_collate
from deepsmlm.generic.utils.processing import TransformSequence
from deepsmlm.neuralfitter.dataset import UnsupervisedDataset, SMLMDatasetOnFly
from deepsmlm.neuralfitter.pre_processing import ROIOffsetRep, N2C
from deepsmlm.neuralfitter.scale_transform import InverseOffsetRescale


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
            for (x_in, ix) in tqdm(self.dataloader):
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

        # gt_fix = int(self.gt.frame_ix.min().item())  # first index
        # gt_lix = int(self.gt.frame_ix.max().item())  # last index
        #
        # gt_frames = self.gt.split_in_frames(gt_fix, gt_lix)
        # pred_frames = self.prediction.split_in_frames(gt_fix, gt_lix)

        self.evaluator.forward(self.prediction, self.gt)


class PredictEvalSimulation(PredictEval):
    def __init__(self, eval_size, prior, simulator, model, post_processor, evaluator=None, param=None, px_size=100.,
                 device='cuda', batch_size=32, multi_frame=True, dataset=None, data_loader=None):
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

        if ((self.param is None) and (self.dataset is None) and (self.dataloader) is None):
            raise ValueError("You need to provide the parameters or you need to provide a dataset and a data loader."
                             "Do the latter if the former fails.")

        self._init_dataset()

    def _init_dataset(self):
        input_preparation = N2C()
        target_generator = TransformSequence.parse([ROIOffsetRep, InverseOffsetRescale], self.param)

        if self.dataset is None:
            self.dataset = SMLMDatasetOnFly(None, self.prior, self.simulator, self.eval_size, input_preparation,
                                            target_generator, None, return_em_tar=False)

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
    from deepsmlm.generic.emitter import *
    from deepsmlm.generic.plotting.frame_coord import *
    from deepsmlm.generic.inout.load_calibration import *
    from deepsmlm.generic.psf_kernel import *
    from deepsmlm.generic.noise import *
    from deepsmlm.generic.utils.data_utils import *
    from deepsmlm.simulation.simulator import *
    from deepsmlm.generic.background import *
    from deepsmlm.simulation.structure_prior import *
    from deepsmlm.neuralfitter.dataset import *
    from deepsmlm.neuralfitter.models.model import *
    from deepsmlm.neuralfitter.models.model_offset import *
    from deepsmlm.neuralfitter.models.model_beta import *
    from deepsmlm.neuralfitter.models.unet_param import *
    from deepsmlm.neuralfitter.pre_processing import *
    from deepsmlm.neuralfitter.post_processing import *
    from deepsmlm.neuralfitter.scale_transform import *
    from deepsmlm.generic.inout.load_save_model import *
    from deepsmlm.neuralfitter.pred_tif import *
    from deepsmlm.evaluation.evaluation import *
    import deepsmlm.generic.utils.processing as processing
    from deepsmlm.neuralfitter.arguments import *
    from deepsmlm.generic.phot_camera import *
    from deepsmlm.evaluation.match_emittersets import GreedyHungarianMatching
    import deepsmlm.generic.inout.write_load_param as wlp

    deepsmlm_root = '/home/lucas/RemoteDeploymentTemp/DeepSMLMv2/'
    os.chdir(deepsmlm_root)

    tifs = 'data/thesis/SMLM_Challenge/T_MT0.N1.HD/sequence-as-stack-MT0.N1.HD-AS-Exp.tif'
    activations = 'data/thesis/SMLM_Challenge/T_MT0.N1.HD/activations.csv'
    model_file = 'network/2019-11-30/dmunet_wd_no_moell_9.pt'
    param_file = 'network/2019-11-30/dmunet_wd_no_moell_param.json'

    param = wlp.load_params(param_file)
    param['Hardware']['num_worker_sim'] = 6
    model = LoadSaveModel(DoubleMUnet.parse(param), None, input_file=model_file).load_init(cuda=True)

    post_processor = processing.TransformSequence.parse([OffsetRescale,
                                                         Offset2Coordinate,
                                                         ConsistencyPostprocessing], param)

    matcher = GreedyHungarianMatching.parse(param)
    segmentation_eval = SegmentationEvaluation(False)
    distance_eval = DistanceEvaluation(print_mode=False)
    batch_ev = BatchEvaluation(matcher, segmentation_eval, distance_eval,
                                          batch_size=param['HyperParameter']['batch_size'],
                                          px_size=torch.tensor(param['Camera']['px_size']))

    predictor = PredictEvalTif(tifs, activations, model, post_processor, batch_ev, None)
    predictor.load_tif()
    predictor.load_csv()
    predictor.gt.convert_em_(factor=torch.tensor([0.01, 0.01, 1.]), shift=torch.tensor([-0.5, -1.5, 0.]), axis=[1, 0, 2])
    predictor.gt.frame_ix -= 1
    # convert photons
    predictor.frames = InputFrameRescale.parse(param).forward(predictor.frames)
    predictor.init_dataset()
    _, output = predictor.forward(True)
    predictor.evaluate()
    print("Done.")


