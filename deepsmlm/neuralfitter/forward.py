import numpy as np
import torch

from deepsmlm.generic.io.load_save_model import LoadSaveModel
from deepsmlm.neuralfitter.dataset import SMLMDataset
from deepsmlm.generic.io.load_save_emitter import MatlabInterface



if __name__ == '__main__':
    deepsmlm_root = os.path.abspath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     os.pardir, os.pardir)) + '/'

    data = SMLMDataset('data/spline_1e1.mat')
    model = load_model(file='network/spline_1e4_no_z.pt')
    model.eval()
    num_examples = 2

    plt_rows = num_examples
    f, axarr = plt.subplots(plt_rows, 5)
    #f, axarr = plt.subplots(plt_rows, 3, gridspec_kw={'wspace':0.025, 'hspace':0.05})
    for i in range(num_examples):

        ran_ix = np.random.randint(data.__len__() - 1)
        print(ran_ix)
        input_image, target, _ = data.__getitem__(ran_ix)
        input_image, target = input_image.unsqueeze(0), target.unsqueeze(0)

        if torch.cuda.is_available():  # model_deep.cuda():
            input_image = input_image.cuda()
        output = model(input_image)

        channel = 0

        for j in range(3):
            axarr[i, j].imshow(input_image[:, j, :, :].squeeze(), cmap='gray')
        imt = axarr[i, 3].imshow(target[:, channel, :, :].squeeze(), cmap='gray')
        imp = axarr[i, 4].imshow(output[:, channel, :, :].detach().numpy().squeeze(), cmap='gray')

        # hide labels
        for j in range(3):
            axarr[i, j].set_xticks([])
            axarr[i, j].set_xticks([])
            axarr[i, j].set_xticklabels([])
            axarr[i, j].set_yticklabels([])
            axarr[i, j].set_aspect('equal')
        if channel == 1:
            plt.colorbar(imt, ax=axarr[i, 3], extend='both')
            imt.set_clim(-1000, 1000)

        axarr[i, 1].set_title('Input')
        axarr[i, 3].set_title('Target')
        axarr[i, 4].set_title('Output')
    plt.show()

    print('Done.')
