import time
import warnings
from functools import partial
from typing import Union, Callable

import torch
from tqdm import tqdm

from .. import dataset
from ...generic import emitter
from ...utils import hardware, frames_io


class Infer:

    def __init__(self, model, ch_in: int, frame_proc, post_proc, device: Union[str, torch.device],
                 batch_size: Union[int, str] = 'auto', num_workers: int = 0, pin_memory: bool = False,
                 forward_cat: Union[str, Callable] = 'emitter'):
        """
        Convenience class for inference.

        Args:
            model: pytorch model
            ch_in: number of input channels
            frame_proc: frame pre-processing pipeline
            post_proc: post-processing pipeline
            device: device where to run inference
            batch_size: batch-size or 'auto' if the batch size should be determined automatically (only use in combination with cuda)
            num_workers: number of workers
            pin_memory: pin memory in dataloader
            forward_cat: method which concatenates the output batches. Can be string or Callable.
            Use 'em' when the post-processor outputs an EmitterSet, or 'frames' when you don't use post-processing or if
            the post-processor outputs frames.
        """

        self.model = model
        self.ch_in = ch_in
        self.batch_size = batch_size
        self.device = device
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.frame_proc = frame_proc
        self.post_proc = post_proc

        self.forward_cat = None
        self._forward_cat_mode = forward_cat

        if str(self.device) == 'cpu' and self.batch_size == 'auto':
            warnings.warn("Automatically determining the batch size does not make sense on cpu device. "
                          "Falling back to reasonable value.")
            self.batch_size = 64

    def forward(self, frames: torch.Tensor) -> emitter.EmitterSet:
        """
        Forward frames through model, pre- and post-processing and output EmitterSet

        Args:
            frames:

        """

        """Move Model"""
        model = self.model.to(self.device)
        model.eval()

        """Form Dataset and Dataloader"""
        ds = dataset.InferenceDataset(frames=frames, frame_proc=self.frame_proc, frame_window=self.ch_in)

        if self.batch_size == 'auto':
            # include safety factor of 20%
            bs = int(0.8 * self.get_max_batch_size(model, ds[0].size(), 1, 512))
        else:
            bs = self.batch_size

        # generate concatenate function here because we need batch size for this
        self.forward_cat = self._setup_forward_cat(self._forward_cat_mode, bs)

        dl = torch.utils.data.DataLoader(dataset=ds, batch_size=bs, shuffle=False, drop_last=False,
                                         num_workers=self.num_workers, pin_memory=self.pin_memory)

        out = []

        with torch.no_grad():
            for sample in tqdm(dl):
                x_in = sample.to(self.device)

                # compute output
                y_out = model(x_in)

                """In post processing we need to make sure that we get a single Emitterset for each batch, 
                so that we can easily concatenate."""
                if self.post_proc is not None:
                    out.append(self.post_proc.forward(y_out))
                else:
                    out.append(y_out.detach().cpu())

        """Cat to single emitterset / frame tensor depending on the specification of the forward_cat attr."""
        out = self.forward_cat(out)

        return out

    def _setup_forward_cat(self, forward_cat, batch_size: int):

        if forward_cat is None:
            return lambda x: x

        elif isinstance(forward_cat, str):

            if forward_cat == 'emitter':
                return partial(emitter.EmitterSet.cat, step_frame_ix=batch_size)

            elif forward_cat == 'frames':
                return partial(torch.cat, dim=0)

        elif callable(forward_cat):
            return forward_cat

        else:
            raise TypeError(f"Specified forward cat method was wrong.")

        raise ValueError(f"Unsupported forward_cat value.")

    @staticmethod
    def get_max_batch_size(model: torch.nn.Module, frame_size: Union[tuple, torch.Size],
                           limit_low: int, limit_high: int):
        """
        Get maximum batch size for inference.

        Args: 
            model: model on correct device
            frame_size: size of frames (without batch dimension)
            limit_low: lower batch size limit
            limit_high: upper batch size limit
        """

        def model_forward_no_grad(x: torch.Tensor):
            """
            Helper function because we need to account for torch.no_grad()
            """
            with torch.no_grad():
                o = model.forward(x)

            return o

        assert next(model.parameters()).is_cuda, \
            "Auto determining the max batch size makes only sense when running on CUDA device."

        return hardware.get_max_batch_size(model_forward_no_grad, frame_size, next(model.parameters()).device,
                                           limit_low, limit_high)


class LiveInfer(Infer):
    def __init__(self,
                 model, ch_in: int, *,
                 stream, time_wait=5, safety_buffer: int = 20,
                 frame_proc=None, post_proc=None,
                 device: Union[str, torch.device] = 'cuda:0' if torch.cuda.is_available() else 'cpu',
                 batch_size: Union[int, str] = 'auto', num_workers: int = 0, pin_memory: bool = False,
                 forward_cat: Union[str, Callable] = 'emitter'):
        """
        Inference from memmory mapped tensor, where the mapped file is possibly live being written to.

        Args:
            model: pytorch model
            ch_in: number of input channels
            stream: output stream. Will typically get emitters (along with starting and stopping index)
            time_wait: wait if length of mapped tensor has not changed
            safety_buffer: buffer distance to end of tensor to avoid conflicts when the file is actively being
            written to
            frame_proc: frame pre-processing pipeline
            post_proc: post-processing pipeline
            device: device where to run inference
            batch_size: batch-size or 'auto' if the batch size should be determined automatically (only use in combination with cuda)
            num_workers: number of workers
            pin_memory: pin memory in dataloader
            forward_cat: method which concatenates the output batches. Can be string or Callable.
            Use 'em' when the post-processor outputs an EmitterSet, or 'frames' when you don't use post-processing or if
            the post-processor outputs frames.
        """

        super().__init__(
            model=model, ch_in=ch_in, frame_proc=frame_proc, post_proc=post_proc,
            device=device, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory,
            forward_cat=forward_cat)

        self._stream = stream
        self._time_wait = time_wait
        self._buffer_length = safety_buffer

    def forward(self, frames: Union[torch.Tensor, frames_io.TiffTensor]):

        n_fitted = 0
        n_waited = 0
        while n_waited <= 2:
            n = len(frames)

            if n_fitted == n - self._buffer_length:
                n_waited += 1
                time.sleep(self._time_wait)  # wait
                continue

            n_2fit = n - self._buffer_length
            out = super().forward(frames[n_fitted:n_2fit])
            self._stream(out, n_fitted, n_2fit)

            n_fitted = n_2fit
            n_waited = 0

        # fit remaining frames
        out = super().forward(frames[n_fitted:n])
        self._stream(out, n_fitted, n)


if __name__ == '__main__':
    import argparse
    import yaml

    import decode.neuralfitter.models
    import decode.utils

    parse = argparse.ArgumentParser(
        description="Inference. This uses the default, suggested implementation. "
                    "For anything else, consult the fitting notebook and make your changes there.")
    parse.add_argument('frame_path', help='Path to the tiff file of the frames')
    parse.add_argument('frame_meta_path', help='Path to the meta of the tiff (i.e. camera parameters)')
    parse.add_argument('model_path', help='Path to the model file')
    parse.add_argument('param_path', help='Path to the parameters of the training')
    parse.add_argument('device', help='Device on which to do inference (e.g. "cpu" or "cuda:0"')
    parse.add_argument('-o', '--online', action='store_true')

    args = parse.parse_args()
    online = args.o

    """Load the model"""
    param = decode.utils.param_io.load_params(args.param_path)

    model = decode.neuralfitter.models.SigmaMUNet.parse(param)
    model = decode.utils.model_io.LoadSaveModel(
        model, input_file=args.model_path, output_file=None).load_init(args.device)

    """Load the frame"""
    if not online:
        frames = decode.utils.frames_io.load_tif(args.frame_path)
    else:
        frames = decode.utils.frames_io.TiffTensor(args.frame_path)

    # load meta
    with open(args.frame_meta_path) as meta:
        meta = yaml.safe_load(meta)

    param = decode.utils.param_io.autofill_dict(meta['Camera'], param.to_dict(), mode_missing='include')
    param = decode.utils.param_io.RecursiveNamespace(**param)

    camera = decode.simulation.camera.Photon2Camera.parse(param)
    camera.device = 'cpu'

    """Prepare Pre and post-processing"""

    """Fit"""

    """Return"""

