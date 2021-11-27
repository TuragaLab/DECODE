import argparse
import yaml

import decode.utils


if __name__ == '__main__':
    parse = argparse.ArgumentParser(
        description="Inference. This uses the default, suggested implementation. "
                    "For anything else, consult the fitting notebook and make your changes there.")
    parse.add_argument('--fit_meta_path', '-p', help='Path to the fit meta file that specifies all '
                                                     'following options in a yaml file')
    args = parse.parse_args()

    """Meta file"""
    if args.fit_meta_path is not None:
        with open(args.fit_meta_path) as f:
            meta = yaml.safe_load(f)

        device = meta['Hardware']['device']
        worker = meta['Hardware']['worker'] if meta['Hardware']['worker'] is not None else 4

        frame_path = meta['Frames']['path']
        frame_meta = meta['Camera']
        frame_range = meta['Frames']['range']

        model_path = meta['Model']['path']
        model_param_path = meta['Model']['param_path']

        output = meta['Output']['path']
    else:
        raise ValueError

    online = False

    # ToDo: This is a massive code duplication of the Fitting Notebook. PLEASE CLEAN UP!

    """Load the model"""
    param = decode.utils.param_io.load_params(model_param_path)

    model = decode.neuralfitter.models.SigmaMUNet.parse(param)
    model = decode.utils.model_io.LoadSaveModel(
        model, input_file=model_path, output_file=None).load_init(device)

    """Load the frame"""
    frames = decode.utils.frames_io.TiffTensor(frame_path)
    frame_range = slice(*frame_range) if frame_range is not None else slice(None)
    if not online:
        frames = frames[frame_range]
    else:
        raise NotImplementedError

    # overwrite camera with meta
    param.Camera = decode.utils.param_io.autofill_dict(frame_meta, param.Camera.to_dict(),
                                                       mode_missing='include')
    param.Camera = decode.utils.param_io.RecursiveNamespace(**param.Camera)

    camera = decode.simulation.camera.Photon2Camera.parse(param)
    camera.device = 'cpu'

    """Prepare Pre and post-processing"""
    frame_proc = [
        decode.neuralfitter.utils.processing.wrap_callable(camera.backward),
        decode.neuralfitter.frame_processing.AutoCenterCrop(8),
        decode.neuralfitter.scale_transform.AmplitudeRescale.parse(param)
    ]
    if param.Camera.mirror_dim is not None:
        frame_proc.insert(2, decode.neuralfitter.frame_processing.Mirror2D(
            dims=param.Camera.mirror_dim))

    # setup frame processing as by the parameter with which the model was trained
    frame_proc = decode.neuralfitter.utils.processing.TransformSequence(frame_proc)

    # determine extent of frame and its dimension after frame_processing
    size_procced = decode.neuralfitter.frame_processing.get_frame_extent(frames.unsqueeze(1).size(),
                                                                         frame_proc.forward)  # frame size after processing
    frame_extent = ((-0.5, size_procced[-2] - 0.5), (-0.5, size_procced[-1] - 0.5))

    # Setup post-processing
    # It's a sequence of backscaling, relative to abs. coord conversion and frame2emitter conversion
    post_proc = decode.neuralfitter.utils.processing.TransformSequence([

        decode.neuralfitter.scale_transform.InverseParamListRescale.parse(param),

        decode.neuralfitter.coord_transform.Offset2Coordinate(xextent=frame_extent[0],
                                                              yextent=frame_extent[1],
                                                              img_shape=size_procced[-2:]),

        decode.neuralfitter.post_processing.SpatialIntegration(raw_th=0.1,
                                                               xy_unit='px',
                                                               px_size=param.Camera.px_size)

    ])

    """Fit"""
    infer = decode.neuralfitter.Infer(model=model, ch_in=param.HyperParameter.channels_in,
                                      frame_proc=frame_proc, post_proc=post_proc,
                                      device=device, num_workers=worker)

    emitter = infer.forward(frames[:])
    emitter.save(output)
    print(f"Fit done and emitters saved to {output}")
