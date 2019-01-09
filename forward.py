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


def plot_frame(tensor):
    img = tensor.squeeze()
    plt.imshow(img, cmap='gray')


if __name__ == '__main__':
    data = SMLMDataset('data/test_32px_1e4.npz', transform=['project01', 'normalise'])
    model = load_model(file='network/trained_32px_1e6_interpoint.pt')
    model.eval()
    num_examples = 2

    plt_rows = num_examples
    f, axarr = plt.subplots(plt_rows, 3)
    #f, axarr = plt.subplots(plt_rows, 3, gridspec_kw={'wspace':0.025, 'hspace':0.05})
    for i in range(num_examples):

        ran_ix = np.random.randint(data.__len__() - 1)
        print(ran_ix)
        input_image, target = data.__getitem__(ran_ix)
        input_image, target = input_image.unsqueeze(0), target.unsqueeze(0)

        if torch.cuda.is_available():  # model_deep.cuda():
            input_image = input_image.cuda()
        output = model(input_image)

        axarr[i, 0].imshow(input_image.squeeze(), cmap='gray')
        axarr[i, 1].imshow(target.squeeze(), cmap='gray')
        axarr[i, 2].imshow(output.detach().numpy().squeeze(), cmap='gray')

        # hide labels
        for j in range(3):
            axarr[i, j].set_xticks([])
            axarr[i, j].set_xticks([])
            axarr[i, j].set_xticklabels([])
            axarr[i, j].set_yticklabels([])
            axarr[i, j].set_aspect('equal')
        axarr[i, 0].set_title('Input')
        axarr[i, 1].set_title('Target')
        axarr[i, 2].set_title('Output')
    plt.show()

    print('Done.')
