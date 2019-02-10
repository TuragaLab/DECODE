

def bump_mse_loss_3d(output, target, kernel_pred, kernel_true, l2=torch.nn.MSELoss(), lz_sc=0.001):

    # call bump_mse_loss on first channel (photons)
    loss_photons = bump_mse_loss(output[:, [0], :, :], target[:, [0], :, :], kernel_pred, kernel_true, l1_sc=1, l2_sc=1)

    output_local_nz = kernel_pred(output[:, [0], :, :]) * kernel_pred(output[:, [1], :, :])
    target_local_nz = kernel_pred(target[:, [0], :, :]) * kernel_pred(target[:, [1], :, :])

    loss_z = l2(output_local_nz, target_local_nz)

    return loss_photons + lz_sc * loss_z


def bump_mse_loss(output, target, kernel_pred, kernel_true=lambda x: x, l1=torch.nn.L1Loss(), l2=torch.nn.MSELoss(), l1_sc=1, l2_sc=1):
    heatmap_pred = kernel_pred(output)
    heatmap_true = kernel_true(target)

    l1_loss = l1(output, torch.zeros_like(target))
    l2_loss = l2(heatmap_pred, heatmap_true)

    return l1_sc * l1_loss + l2_sc * l2_loss  # + 10**(-2) * loss_num


def interpoint_loss(input, target, threshold=500):
    def expanded_pairwise_distances(x, y=None):
        '''
        Taken from https://discuss.pytorch.org/t/efficient-distance-matrix-computation/9065

        Input: x is a Nxd matrix
               y is an optional Mxd matirx
        Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
                if y is not given then use 'y=x'.
        i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
        '''
        if y is not None:
            differences = x.unsqueeze(1) - y.unsqueeze(0)
        else:
            differences = x.unsqueeze(1) - x.unsqueeze(0)
        distances = torch.sum(differences * differences, -1)
        return distances

    interpoint_dist = expanded_pairwise_distances(
        (input >= threshold).nonzero(), target.nonzero())
    # return distance to closest target point
    return interpoint_dist.min(1)[0].sum() / input.__len__()


def inverse_intens(output, target):
    pass


def num_active_emitter_loss(input, target, threshold=0.15):
    input_f = input.view(*input.shape[:2], -1)
    target_f = target.view(*target.shape[:2], -1)

    num_true_emitters = torch.sum(target_f > threshold * target_f.max(), 2)
    num_pred_emitters = torch.sum(input_f > threshold * input_f.max(), 2)

    loss = ((num_pred_emitters - num_true_emitters)
            ** 2).sum() / input.__len__()
    return loss.type(torch.FloatTensor)


def bump_mse_loss_3d(output, target, kernel_pred, kernel_true, l2=torch.nn.MSELoss(), lz_sc=0.001):

    # call bump_mse_loss on first channel (photons)
    loss_photons = bump_mse_loss(output[:, [0], :, :], target[:, [0], :, :], kernel_pred, kernel_true, l1_sc=1, l2_sc=1)

    output_local_nz = kernel_pred(output[:, [0], :, :]) * kernel_pred(output[:, [1], :, :])
    target_local_nz = kernel_pred(target[:, [0], :, :]) * kernel_pred(target[:, [1], :, :])

    loss_z = l2(output_local_nz, target_local_nz)

    return loss_photons + lz_sc * loss_z
