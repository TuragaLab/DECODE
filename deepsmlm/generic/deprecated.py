def transformation_of_a_dataset():
    if self.transform is not None:
        if 'project01' in self.transform:
            self.images = project01(self.images)
            # self.images_hr = project01(self.images_hr)
        if 'normalise' in self.transform:
            mean = self.images.mean()
            std = self.images.std()

            torch.save([mean, std], inputfile[:-3] + '_normalisation.pt')

            self.images = normalise(self.images, mean, std)

        if 'test_set_norm_from_train' in self.transform:
            [mean, std] = torch.load(transform_vars)
            self.images = normalise(self.images, mean, std)


def weighted_avg_std(values, weights):
    """
    https://stackoverflow.com/questions/2413522/weighted-standard-deviation-in-numpy
    Return the weighted average and standard deviation.
    values, weights -- Numpy ndarrays with the same shape.
    """
    average = np.average(values, weights=weights)
    variance = np.average((values-average)**2, weights=weights)
    return average, np.sqrt(variance)


        # if shotnoise:
        #     return torch.distributions.poisson.Poisson(gaussCdf).sample().unsqueeze(0)
        # else:
        #     return gaussCdf.unsqueeze(0)


def gaussian_convolution(pos,
                    sig=(2, 2), photon_count=1000, img_shape=np.array([64, 64]), up_fact=10):
    """
    Function to compute the gaussian PSF with a convolution. Grid based.

    :param pos:
    :param sig:
    :param photon_count:
    :param img_shape:
    :param up_fact:

    :return:
    """

    # use a finer grid, transform coordinates from px to fine-px grid
    _pos = (0.5 + pos) * up_fact # not yet understood why it's off by half a pixel
    _sig = up_fact * sig
    img_shape_hr = np.array(img_shape) * up_fact

    # compute kernel. is this the cdf?
    kernel = np.outer(signal.gaussian(np.ceil(10 * _sig[0]), _sig[0]),
                      signal.gaussian(np.ceil(10 * _sig[1]), _sig[1]))
    kernel = kernel / np.sum(kernel)

    img_hr = convolutional_sample(_pos, photon_count, kernel, img_shape_hr)

    # downsample
    img_lr = block_reduce(img_hr, block_size=(up_fact, up_fact), func=np.sum)

    raise NotImplementedError('Image vector has wrong dimensionality. Need to add singleton dimension for channels.')

    return img_lr


def convolutional_sample(pos, p_count, kernel, img_shape):  # this convolution could be done with pytorch on the gpu
    """
    Function to convolute an image made up ouf single positions with a kernel

    :param pos:
    :param p_count:
    :param kernel:
    :param img_shape:
    :return:
    """
    # round positions to integer
    pos = np.round(pos).astype(int)
    pos = np.clip(pos, 0, img_shape[0] - 1)

    img = np.zeros((img_shape[0], img_shape[1]))
    img[pos[:, 1], pos[:, 0]] = p_count  # this effectively is the centre of the px
    img = np.clip(signal.fftconvolve(img, kernel, mode='same'), 0, None)
    img = np.random.poisson(img)  # inherent poisson noise

    raise NotImplementedError('Image vector has wrong dimensionality. Need to add singleton dimension for channels.')

    return img
