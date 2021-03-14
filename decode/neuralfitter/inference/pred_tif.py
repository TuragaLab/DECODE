import csv
from abc import ABC, abstractmethod  # abstract class
from deprecated import deprecated

import torch
import torch.utils
from tqdm import tqdm

import decode.generic.emitter as em
from decode.neuralfitter.dataset import InferenceDataset
from decode.neuralfitter.utils.dataloader_customs import smlm_collate


@deprecated(reason="Depr. in favour of inference.Infer", version="0.1.dev")
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
                x_in = sample.to(self.device)

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
                x_in = sample.to(self.device)

                # compute output
                output = self.model(x_in)
                raw_frames.append(output.detach().cpu())

        # put model back to cpu
        self.model = self.model.to(torch.device('cpu'))

        raw_frames = torch.cat(raw_frames, 0)
        return raw_frames

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


@deprecated(reason="Depr. in favour of inference.Infer", version="0.1.dev")
class PredictEvalSimulation(PredictEval):
    def __init__(self, eval_size, prior, simulator, model, post_processor, evaluator=None, param=None,
                 device='cuda', batch_size=32, input_preparation=None, multi_frame=True, dataset=None,
                 data_loader=None):

        super().__init__(model, post_processor, evaluator, batch_size, device)

        self.eval_size = eval_size
        self.prior = prior
        self.simulator = simulator
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

        self.input_preparation = input_preparation

        self._init_dataset()

    def _init_dataset(self):

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


@deprecated(reason="Depr. in favour of inference.Infer", version="0.1.dev")
class PredictEvalTif(PredictEval):
    def __init__(self, tif_stack, activations, model, post_processor, frame_proc, evaluator=None, device='cuda',
                 batch_size=32, frame_window: int = 3):

        super().__init__(model=model,
                         post_processor=post_processor,
                         evaluator=evaluator,
                         batch_size=batch_size,
                         device=device)

        self.tif_stack = tif_stack
        self.activation_file = activations
        self.frame_window = frame_window
        self.frame_proc = frame_proc

        self.prediction = None
        self.frames = None
        self.dataset = None
        self.dataloader = None

    @staticmethod
    def load_csv(activation_file, verbose=False):

        if activation_file is None:
            print("WARNING: No activations loaded since file not specified; i.e. there is no ground truth.")
            return

        # read csv
        with open(activation_file) as csv_file:
            csv_reader = csv.reader(csv_file)
            line_count = 0
            id_frame_xyz_camval = []
            for row in csv_reader:
                if verbose and line_count == 0:
                    print(row)
                elif line_count >= 1:
                    id_frame_xyz_camval.append(torch.tensor(
                        (float(row[0]), float(row[1]), float(row[2]), float(row[3]), float(row[4]), float(row[5]))))
                line_count += 1

        id_frame_xyz_camval = torch.stack(id_frame_xyz_camval, 0)

        gt = em.EmitterSet(xyz=id_frame_xyz_camval[:, 2:5], frame_ix=id_frame_xyz_camval[:, 1].long(),
                           phot=id_frame_xyz_camval[:, -1], id=id_frame_xyz_camval[:, 0].long())
        gt.sort_by_frame_()

        return gt

    def load_tif_csv(self):
        self.frames = self.load_tif(self.tif_stack)
        self.gt = self.load_csv(self.activation_file)

    def init_dataset(self, frames=None):
        """
        Initiliase the dataset. Usually by preloaded frames but you can overwrite.
        :param frames: N C(=1) H W
        :return:
        """
        if frames is None:
            frames = self.frames

        self.dataset = InferenceDataset(frames=frames,
                                        frame_window=self.frame_window,
                                        frame_proc=self.frame_proc)
        self.dataloader = torch.utils.data.DataLoader(self.dataset,
                                                      batch_size=self.batch_size, shuffle=False,
                                                      num_workers=8, pin_memory=True)
