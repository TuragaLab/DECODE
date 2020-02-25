from abc import ABC, abstractmethod  # abstract class

import pathlib
import torch
import torch.utils
import tifffile
import csv
from tqdm import tqdm

import deepsmlm.generic.emitter as em
from deepsmlm.neuralfitter.utils.pytorch_customs import smlm_collate
from deepsmlm.generic.utils.processing import TransformSequence
from deepsmlm.neuralfitter.dataset import UnsupervisedDataset, SMLMDatasetOnFly, SMLMDatasetOneTimer
from deepsmlm.neuralfitter.pre_processing import N2C
from deepsmlm.neuralfitter.target_generator import ROIOffsetRep
from deepsmlm.neuralfitter.scale_transform import InverseOffsetRescale, AmplitudeRescale


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
            for sample in tqdm(self.dataloader):
                x_in = sample[0]
                x_in = x_in.to(self.device)

                # compute output
                output = self.model(x_in)
                if output_raw:
                    raw_frames.append(output.detach().cpu())
                """In post processing we need to make sure that we get a single Emitterset for each batch, 
                so that we can easily concatenate."""
                em_outs.append(self.post_processor.forward(output))

        # put model back to cpu
        self.model = self.model.to(torch.device('cpu'))

        em_merged = em.EmitterSet.cat(em_outs, step_frame_ix=self.batch_size)
        self.prediction = em_merged
        if output_raw:
            raw_frames = torch.cat(raw_frames, 0)
            return self.prediction, raw_frames
        else:
            return self.prediction

    def forward_raw(self):
        """
        Forwards the data through the model but without post-processing

        Returns: raw_frames (torch.Tensor)

        """

        """Eval mode."""
        raw_frames = []
        self.model.to(self.device)

        """Eval mode and no grad."""
        self.model.eval()
        with torch.no_grad():
            for sample in tqdm(self.dataloader):
                x_in = sample[0]
                x_in = x_in.to(self.device)

                # compute output
                output = self.model(x_in)
                raw_frames.append(output.detach().cpu())

        # put model back to cpu
        self.model = self.model.to(torch.device('cpu'))

        raw_frames = torch.cat(raw_frames, 0)
        return raw_frames

    # def forward_post(self):
    #     """
    #     Forwards raw frames ty the post processor
    #     Returns:
    #
    #     """

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

        if (self.param is None) and (self.dataset is None) and (self.dataloader) is None:
            raise ValueError("You need to provide the parameters or you need to provide a dataset and a data loader."
                             "Do the latter if the former fails.")

        if self.input_preparation is None:
            self.input_preparation = TransformSequence([
                N2C(),
                AmplitudeRescale.parse(param)
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
        super().__init__(model=model,
                         post_processor=post_processor,
                         evaluator=evaluator,
                         batch_size=batch_size,
                         device=device)

        self.tif_stack = tif_stack
        self.activation_file = activations
        self.px_size = px_size
        self.multi_frame = multi_frame

        self.prediction = None
        self.frames = None
        self.dataset = None
        self.dataloader = None

    def load_tif(self, no_init=False):
        """
        Reads the tif(f) files. When a folder is specified, potentially multiple files are loaded.
        Which are stacked into a new first axis.
        Make sure that if you provide multiple files (i.e. a folder) sorting gives the correct order. Otherwise we can
        not guarantee anything.

        Args:
            no_init: (bool) do not init dataset. useful if you want to manipulate your data first

        Returns:

        """
        p = pathlib.Path(self.tif_stack)

        # if dir, load multiple files and stack them if more than one found
        if p.is_dir():
            print("Path to folder of tifs specified. Traversing through the directory.")

            file_list = sorted(p.glob('*.tif*'))  # load .tif or .tiff
            frames = []
            for f in tqdm(file_list):
                frames.append(torch.from_numpy(tifffile.imread(str(f)).astype('float32')))

            print("Tiffs successfully read.")
            if frames.__len__() >= 2:
                print("Multiple tiffs found. Stacking them ...")
                frames = torch.stack(frames, 0)
            else:
                frames = frames[0]

        else:
            im = tifffile.imread(self.tif_stack)
            print("Tiff successfully read.")
            frames = torch.from_numpy(im.astype('float32'))

        if frames.squeeze().ndim <= 2:
            raise ValueError("Tif seems to be of wrong dimension or could only find a single frame.")

        frames.unsqueeze_(1)

        self.frames = frames
        if not no_init:
            self.init_dataset()

    def init_dataset(self, frames=None):
        """
        Initiliase the dataset. Usually by preloaded frames but you can overwrite.
        :param frames: N C(=1) H W
        :return:
        """
        if frames is None:
            frames = self.frames

        self.dataset = UnsupervisedDataset(None, frames=frames,
                                           multi_frame_output=self.multi_frame)
        self.dataloader = torch.utils.data.DataLoader(self.dataset,
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
        gt.sort_by_frame_()

        return gt

    def load_tif_csv(self):
        self.load_tif()
        self.load_csv_()
