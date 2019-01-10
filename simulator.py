import itertools as iter
import numpy as np
import matplotlib.pyplot as plt
import functools
import torch
from multiprocessing.dummy import Pool as ThreadPool

from psf_kernel import gaussian_expect, gaussian_convolution, noise_psf, delta_psf


class Simulation:
    """
    A class representing a smlm simulation
    """
    def __init__(self, em_mat=None, cont_mat=None, img_size=(64, 64), img_size_hr=(64,64), upscale=1,
                 background=None, psf=None, psf_hr=None, performance=None, poolsize=4):
        """
        Initialise a simulation
        :param emitter_mat: matrix of emitters. N x 6. 0-2 col: xyz, 3 photon, 4: frameix, 5 emit id
        :param img_size: tuple of image dimension
        :param img_size_hr: tuple of high_resolution image dimension (i.e. image with delta function psf)
        :param background: function to generate background, comprises a fct for pre capture bg and post bg
                            input: image
        :param psf: psf function
        :param psf_hr: psf function for high resolution image (e.g. delta function psf)

        :param performance: function to evaluate performance of prediction

        class own
        :param predictions: array of instances of predicted emitters
        :param image: camera image
        """

        self.emitter_mat = em_mat
        self.contaminator_mat = cont_mat

        self.img_size = img_size
        self.img_size_hr = img_size_hr
        self.upscale = upscale
        self.image = None
        self.image_hr = None

        self.predictions = None

        self.background = background
        self.psf = psf
        self.psf_hr = psf_hr
        self.performance = performance
        self.poolsize = poolsize

    @property
    def num_emitters(self):
        return self.emitter_mat.shape[0]

    @property
    def num_frames(self):
        em_matrix = self.get_emitter_matrix('all')
        return int(torch.max(em_matrix[:, 4]) + 1)

    def camera_image(self, psf=None, bg=True, upscale=1, emitter_kind='all'):
        # get emitter matrix
        em_mat = self.get_emitter_matrix(kind=emitter_kind)

        pool = ThreadPool(self.poolsize)
        frame_list = pool.starmap(self.get_frame_wrapper, zip(list(range(self.num_frames)),
                                                     iter.repeat(em_mat),
                                                     iter.repeat(psf),
                                                     iter.repeat(bg)))
        img = torch.stack(frame_list, dim=2)  # np.moveaxis(np.asarray(frame_list), 0, -1)

        return img.type(torch.int16)

    def directed_distance(em_mat, px_positions, image_shape):
        em_mat = torch.from_numpy(em_mat)
        image_shape = torch.from_numpy(image_shape)
        # get index of emitter which is closest, remember the order
        ix_closest_emitter = pairwise_distances(px_positions, em_mat).min(1)[1]

        # place cordinates of closest emitter
        directed_distance = (em_mat[ix_closest_emitter, :] - px_positions).view(em_mat.shape[1], *image_shape)

        return directed_distance

    def get_frame_wrapper(self, ix, em_mat, psf, bg=True):
        em_in_frame = em_mat[em_mat[:, 4] == ix, :]
        return self.get_frame(em_in_frame[:, :3], em_in_frame[:, 3], psf=psf, bg=bg)

    def get_frame(self, pos, phot, psf, bg=True):
        if psf is None:
            psf = self.psf

        frame = psf(pos, phot)
        if bg:
            frame = self.background(frame)
        return frame

    @functools.lru_cache()
    def get_emitter_matrix(self, kind='emitter'):
        """
        :param emitters:
        :return: Matrix number of emitters x 5. (x,y,z, photon count, frameix)
        """
        if kind == 'all':
            return torch.cat([self.emitter_mat, self.contaminator_mat], dim=0)
        elif kind == 'emitter':
            return self.emitter_mat
        elif kind == 'contaminator':
            return self.contaminator_mat
        else:
            raise ValueError('Not supported kind.')

    def get_emitter_matrix_frame(self, ix, kind=None):
        em_mat = self.get_emitter_matrix(kind=kind)
        return em_mat[em_mat[:, 4] == ix, :4]

    @functools.lru_cache()
    def get_extent(self, img_size):
        return [-0.5, img_size[0]/self.upscale - 0.5, img_size[1]/self.upscale - 0.5, -0.5]

    @property
    def simulation_extent(self):
        return self.get_extent(self.img_size)

    @property
    def simulation_extent_hr(self):
        return self.get_extent(self.img_size_hr)

    def plot_frame(self, ix, image=None, crosses=True):
        if image is None:
            image = self.image

        plt.imshow(image[:, :, ix].numpy(), cmap='gray', extent=self.get_extent(image.shape[:2]))

        ground_truth = self.get_emitter_matrix_frame(ix, kind='emitter')
        if crosses is True:
            plt.plot(ground_truth[:, 0].numpy(), ground_truth[:, 1].numpy(), 'rx')

        if self.predictions is not None:
            plt.plot(localisation[:, 0].numpy(), localisation[:, 1].numpy(), 'bo')

    def write_to_binary(self, outfile):

        np.savez_compressed(outfile, frames=self.image, frames_hr=self.image_hr, emitters=self.get_emitter_matrix(), len=self.num_frames)


class Emitter:
    """
    Class representing a single emitter
    """
    def __init__(self, position, photons, frames, is_contaminator=False):
        """

        :param position: x,y,z emitters position as numpy array, size 3
        :param photons: number of photons per frame, size: num_frames
        :param frames: frame indices, size: num_frames

        :param is_contaminator:
        """
        self.position = position
        self.photons = photons
        self.frames = frames

        self.contaminator = is_contaminator

    def return_matrix(self):
        num_frames = self.frames.__len__()
        p = torch.zeros((num_frames, 5))  # frames x (x,y,z,photons,frame_ix)

        p[:, :3] = np.repeat(torch.unsqueeze(self.position, 0), num_frames, axis=0)  # change!
        p[:, 3] = self.photons
        p[:, 4] = self.frames
        return p


# https://discuss.pytorch.org/t/efficient-distance-matrix-computation/9065
def pairwise_distances(x, y=None):  # not numerically stable but fast
    '''
    Input: x is a Nxd matrix
           y is an optional Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
            if y is not given then use 'y=x'.
    i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
    '''
    x_norm = (x**2).sum(1).view(-1, 1)
    if y is not None:
        y_norm = (y**2).sum(1).view(1, -1)
    else:
        y = x
        y_norm = x_norm.view(1, -1)

    dist = x_norm + y_norm - 2.0 * torch.mm(x, torch.transpose(y, 0, 1))
    return dist


# https://discuss.pytorch.org/t/efficient-distance-matrix-computation/9065
def expanded_pairwise_distances(x, y=None):  # numerically stable but slow
    '''
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


def upscale(img, scale=1):  # either 2D or 3D batch of 2D
    if img.dim() == 3:
        output = np.kron(img.numpy(), np.ones((scale, scale, 1)))
    else:
        output = np.kron(img.numpy(), np.ones((scale, scale)))
    return torch.from_numpy(output)


def dist_phot_lifetime(start_frame, lifetime, photon_count):
    # return photon count per frame index
    raise RuntimeError("Not yet supported.")


def random_emitters(emitter_per_frame, frames, lifetime, img_size, cont_radius=3):

    if lifetime is None:  # assume 1 frame emitters
        num_emitters = emitter_per_frame * frames
        positions = torch.rand(num_emitters, 3) * (img_size[0] + 2 * cont_radius) - cont_radius  # np.random.uniform(-cont_radius, img_size[0] + cont_radius, (num_emitters, 3))  # place emitters entirely randomly
        start_frame = torch.randint(0, frames, (num_emitters, 1))  # np.random.randint(0, frames, (num_emitters, 1))  # start on state is distributed uniformly
        lifetime_per_emitter = 1

        emit_id =  torch.arange(0, num_emitters).unsqueeze(0).transpose(0, 1)  #  np.expand_dims(np.mgrid[0:num_emitters], 1)
    else:  # prototype
        raise NotImplementedError
        num_emitters = torch.round(emitter_per_frame * frames / lifetime)  # roughly
        positions = torch.uniform(num_emitters, 3) * (img_size[0] + 2 * cont_radius) - cont_radius  # place emitters entirely randomly
        # start_frame = np.random.uniform(0, frames, (num_emitters, 1))  # start on state is distributed uniformly
        lifetime_per_emitter = torch.zeros(num_emitters).exponential(lifetime)  # np.random.exponential(lifetime, num_emitters)  # lifetime drawn from exponential dist

    photon_count = torch.randint(800, 4000, (num_emitters, 1))  # np.random.randint(800, 4000, (num_emitters, 1))

    emit_all = torch.cat((positions, photon_count.float(), start_frame.float(), emit_id.float()), 1)  # np.concatenate([positions, photon_count, start_frame, emit_id], axis=1)
    is_emit = torch.mul((emit_all[:, :2] >= 0).all(1), (emit_all[:, :2] <= img_size[0] - 1).all(1))  # np.multiply(np.all(emit_all[:, :2] >= 0, 1), np.all(emit_all[:, :2] <= img_size[0] - 1, 1))
    is_cont = ~is_emit

    if img_size[0] != img_size[1]:
        raise NotImplementedError("Image must be square at the moment because otherwise the following doesn't work.")

    emit_mat, cont_mat = emit_all[is_emit, :], emit_all[is_cont, :]
    return emit_mat, cont_mat


if __name__ == '__main__':
    binary_path = 'data/data_32px_up8_1e1_test.npz'

    image_size = (32, 32)
    upscale_factor = 8
    image_size_hr = (image_size[0] * upscale_factor, image_size[1] * upscale_factor)
    emitter_p_frame = 15
    total_frames = 10
    bg_value = 10
    sigma = torch.tensor([1.5, 1.5])

    _bg = lambda img: noise_psf(img, bg_poisson=bg_value)
    _psf = lambda pos, phot: gaussian_expect(pos, sigma, phot, img_shape=image_size)
    _psf_hr = lambda pos, phot: delta_psf(pos, phot, img_shape=image_size_hr,
                                          xextent=torch.tensor([0, image_size[0]], dtype=torch.float),
                                          yextent=torch.tensor([0, image_size[1]], dtype=torch.float))

    emit, cont = random_emitters(emitter_p_frame, total_frames, None, image_size, 3)
    sim = Simulation(emit, cont, img_size=image_size, upscale=upscale_factor,
                     background=_bg, psf=_psf, psf_hr=_psf_hr, poolsize=10)
    sim.image = upscale(sim.camera_image(upscale=1), sim.upscale)  # don't upscale before convolution, scale final image
    sim.image_hr = sim.camera_image(psf=sim.psf_hr, bg=False, upscale=upscale_factor, emitter_kind='emitter')  # work on hr, no need to upscale
    sim.write_to_binary(binary_path)

    print("Generating samples done. Filename: {}".format(binary_path))

    plt.subplot(121)
    sim.plot_frame(0, sim.image_hr, False)
    plt.subplot(122)
    sim.plot_frame(0, sim.image, True)
    plt.show()
    print("Done")
