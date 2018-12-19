import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.autograd import Variable
import matplotlib.pyplot as plt

from model import DeepSLMN
from train import SMLMDataset, load_model


if __name__ == '__main__':
    data = SMLMDataset('data_32px_xlarge.npz')
    model = load_model(file='network/net_14.pt')
    model.eval()
    num_examples = 2

    plt_rows = num_examples
    f, axarr = plt.subplots(plt_rows, 3)
    #f, axarr = plt.subplots(plt_rows, 3, gridspec_kw={'wspace':0.025, 'hspace':0.05})
    for i in range(num_examples):

        ix = np.random.randint(0, data.__len__())

        input = torch.unsqueeze(data.__getitem__(ix)[0], 0)
        if torch.cuda.is_available():  # model_deep.cuda():
            input = input.cuda()
        out = model(input)

        in_np = np.squeeze(data.__getitem__(ix)[0].numpy())
        ground_truth = np.squeeze(data.__getitem__(ix)[1].numpy())
        out_np = out.data.cpu().numpy()
        out_np = out_np.squeeze()

        axarr[i, 0].imshow(in_np, cmap='gray')
        axarr[i, 1].imshow(ground_truth, cmap='gray')
        axarr[i, 2].imshow(out_np, cmap='gray')

        # hide labels
        for j in range(3):
            axarr[i, j].set_xticks([])
            axarr[i, j].set_xticks([])
            axarr[i, j].set_xticklabels([])
            axarr[i, j].set_yticklabels([])
            axarr[i, j].set_aspect('equal')
    plt.show()

    print('Done.')
