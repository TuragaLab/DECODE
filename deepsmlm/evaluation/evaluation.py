from deepsmlm.evaluation.metric_library import pos_neg_emitters, precision_recall, rmse_mad


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


class SegmentationEvaluation:
    """
    Evaluate performance on finding the right emitters.
    """
    def __init__(self, distance_threshold=1, matching_algorithm=pos_neg_emitters):
        """

        :param distance_threshold: (float) to consider two points as close enough
        :param matching_algorithm: algorithm to match two emittersets.
            First three outputs must be true positives, false positives, false negatives
        """
        self.matching_algo = matching_algorithm
        self.matching_param = (distance_threshold,)

    def forward_frame(self, output, target):
        """
        Run the complete segmentation evaluation for two arbitrary emittersets.
        This disregards the frame_ix

        :param output: (instance of emitterset)
        :param target:  (instance of emitterset)
        :return: several metrics
        """

        output = self.matching_algo(output, target, *self.matching_param)
        tp, fp, fn = output[0], output[1], output[2]

        prec, rec = precision_recall(tp.num_emitter, fp.num_emitter, fn.num_emitter)

        return tp.num_emitter, fp.num_emitter, fn.num_emitter, prec, rec


class DistanceEvaluation:
    """
    Evaluate performance on how precise we are.
    """
    def __init__(self):
        pass

    def forward(self, output, target):
        """

        :param output: (instance of Emitterset)
        :param target: (instance of Emitterset)
        :return:
        """

        rmse_vol, rmse_lat, rmse_axial, mad_vol, mad_lat, mad_axial = rmse_mad(target, output)

        return rmse_vol, rmse_lat, rmse_axial, mad_vol, mad_lat, mad_axial


