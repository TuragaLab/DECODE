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

        gt_fix = int(self.gt.frame_ix.min().item())  # first index
        gt_lix = int(self.gt.frame_ix.max().item())  # last index

        gt_frames = self.gt.split_in_frames(gt_fix, gt_lix)
        pred_frames = self.prediction.split_in_frames(gt_fix, gt_lix)

        self.evaluator.forward(pred_frames, gt_frames)


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
                                    target_generator,
                                    None, static=True, lifetime=self.param['HyperParameter']['ds_lifetime'],
                                    return_em_tar=False)

        if self.dataloader is None:
            self.dataloader = torch.utils.data.DataLoader(self.dataset,
                                                          batch_size=self.param['HyperParameter']['batch_size'],
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
        self._init_dataset()

    def _init_dataset(self):

        ds = UnsupervisedDataset(((-0.5, 63.5), (-0.5, 63.5), (-750., 750.)), frames=self.frames,
                                 multi_frame_output=self.multi_frame)
        self.dataloader = torch.utils.data.DataLoader(ds,
                                                      batch_size=self.batch_size, shuffle=False,
                                                      num_workers=8, pin_memory=False)

    def load_csv(self):

        if self.activation_file is None:
            print("WARNING: No activations loaded since file not specified; i.e. there is no ground truth.")
            return

        # read csv
        with open(self.activation_file) as csv_file:
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

        # nm to px
        gt.convert_em_(factor=torch.tensor([1/self.px_size, 1/self.px_size, 1.]))
        self.gt = gt

    def load_tif_csv(self):
        self.load_tif()
        self.load_csv()
