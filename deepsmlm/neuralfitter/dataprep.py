import numpy as np
import matplotlib.pyplot as plt
import torch

from ..simulator.emittergenerator import EmitterSet


def load_temp_binary(input_file):

    """Save template: np.savez('output.npz', xyz=xyz, phot=phot, frame_indx=frame_indx, id=id)"""
    data = np.load(input_file)

    return EmitterSet(data['xyz'], data['phot'], data['frame_indx'], data['id'])


def load_binary(input_file):
    """
    Load a binary which is compatible with the current standard of processing. Return to Dataset class.
    That means:
        frames is a N x C(1) X H X W tensor.


    :param input_file:    input binary
    :return:    images as torch tensor N, C, H, W; emitters as list of dicts
    """

    if input_file[-3:] in ('.np', 'npz'):

        bin = np.load(input_file)
        img_frames = np.ascontiguousarray(bin['img_frames'], dtype=np.float32)

        xyz = bin['xyz_ppn']
        phot = bin['phot']
        frame_indx = bin['frame_indx']

        em = [None] * xyz.__len__()
        for i in range(xyz.__len__()):
            em[i] = EmitterSet(xyz[i], phot, frame_indx, None)

    elif input_file[-3:] == '.pt':
        raise NotImplementedError
        # bin = torch.load(input_file)

    else:
        raise ValueError('Datatype not supported.')

    return torch.from_numpy(frames), torch.from_numpy(em_mat)


def project01(img):
    # 4d
    img = img.contiguous()
    img_flat = img.view(img.shape[0], img.shape[1], -1)
    img_min = img_flat.min(2, keepdim=True)[0]
    img_max = img_flat.max(2, keepdim=True)[0]

    img_flat_norm = (img_flat - img_min) / (img_max - img_min)
    return img_flat_norm.view(img.shape[0], img.shape[1], img.shape[2], img.shape[3])


def get_outputsize(input_size, model):
    input = torch.randn(1, 1, input_size, input_size)
    return model.forward(input).size()


if __name__ == '__main__':
    frames, em_mat = load_binary('data/spline_1e4.mat')
    target = generate_3d_hr_target(em_mat, np.array([32, 32]), 8, np.array([-0.5, 31.5]), np.array([-0.5, 31.5]))

    plt.figure()
    plt.subplot(121)
    plt.imshow(target[0, 0, :, :])
    plt.subplot(122)
    plt.imshow(target[0, 1, :, :])
    plt.show()
