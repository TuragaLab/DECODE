import numpy as np
import matplotlib.pyplot as plt
import functools

import time

from psf_kernel import gaussian_expect, gaussian_convolution, noise_psf, delta_psf


class Simulation:
    """
    A class representing a smlm simulation
    """
    def __init__(self, emitters=None, contaminators=None, img_size=(64, 64), img_size_hr=(64,64), upscale=1,
                 background=None, psf=None, psf_hr=None, performance=None):
        """
        Initialise a simulation
        :param emitters: array of instances of emitters
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

        self.emitters = emitters
        self.contaminators = contaminators
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

    @property
    def num_emitters(self):
        return self.emitters.__len__()

    @property
    def num_frames(self):
        em_matrix = self.get_emitter_matrix('all')
        return np.max(em_matrix[:, 4]).astype('uint16') + 1

    def camera_image(self, psf=None, bg=True, upscale=1):
        # get emitter matrix
        em_mat = self.get_emitter_matrix(kind='all')

        img = np.zeros((self.img_size[0] * upscale, self.img_size[1] * upscale, self.num_frames))
        for i in range(self.num_frames):
            # get emitters in current frame
            em_in_frame = em_mat[em_mat[:, 4] == i, :]
            frame = self.get_frame(em_in_frame[:, :3], em_in_frame[:, 3], psf=psf, bg=bg)
            img[:, :, i] = frame
        return img.astype('uint16')

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
            emitters = self.emitters + self.contaminators
        elif kind == 'emitter':
            emitters = self.emitters
        elif kind == 'contaminator':
            emitters = self.contaminators
        else:
            raise ValueError('Not supported kind.')

        em_mat = np.zeros((emitters.__len__(), 5))
        for i, emitter in enumerate(emitters):
            em_mat[i, :] = emitter.return_matrix()
        return em_mat

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

        plt.imshow(image[:, :, ix], cmap='gray', extent=self.get_extent(image.shape[:2]))

        ground_truth = self.get_emitter_matrix_frame(ix, kind='emitter')
        if crosses is True:
            plt.plot(ground_truth[:, 0], ground_truth[:, 1], 'rx')

        if self.predictions is not None:
            plt.plot(localisation[:, 0], localisation[:, 1], 'bo')

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
        p = np.zeros((num_frames, 5))  # frames x (x,y,z,photons,frame_ix)

        p[:, :3] = np.repeat(np.expand_dims(self.position, axis=0), num_frames, axis=0)
        p[:, 3] = self.photons
        p[:, 4] = self.frames
        return p


def upscale(img, scale=1):  # either 2D or 3D batch of 2D
    if img.ndim == 3:
        return np.kron(img, np.ones((scale, scale, 1)))
    else:
        return np.kron(img, np.ones((scale, scale)))


def dist_phot_lifetime(start_frame, lifetime, photon_count):
    # return photon count per frame index
    raise RuntimeError("Not yet supported.")


def random_emitters(emitter_per_frame, frames, lifetime, img_size, cont_radius=3):

    if lifetime is None:  # assume 1 frame emitters
        num_emitters = emitter_per_frame * frames
        positions = np.random.uniform(-cont_radius, img_size[0] + cont_radius, (num_emitters, 3))  # place emitters entirely randomly
        start_frame = np.random.randint(0, frames, num_emitters)  # start on state is distributed uniformly
        lifetime_per_emitter = 1

    else:  # prototype
        raise NotImplementedError
        num_emitters = np.round(emitter_per_frame * frames / lifetime)  # roughly
        positions = np.random.uniform(-cont_radius, img_size[0] + cont_radius,
                                      (num_emitters, 3))  # place emitters entirely randomly
        start_frame = np.random.uniform(0, frames, num_emitters)  # start on state is distributed uniformly
        lifetime_per_emitter = np.random.exponential(lifetime, num_emitters)  # lifetime drawn from exponential dist

    photon_count = np.random.randint(800, 4000, num_emitters)

    emit = []
    cont = []
    for i in range(num_emitters):
        if np.any(positions[i, :2] < 0) or np.any(positions[i, :2] > img_size[0]):
            cont.append(Emitter(positions[i, :], np.array([photon_count[i]]), np.array([start_frame[i]]), True))
        else:
            emit.append(Emitter(positions[i, :], np.array([photon_count[i]]), np.array([start_frame[i]])))

    return emit, cont


if __name__ == '__main__':
    binary_path = 'data/check.npz'

    image_size = (32, 32)
    upscale_factor = 1
    image_size_hr = (image_size[0] * upscale_factor, image_size[1] * upscale_factor)
    emitter_p_frame = 15
    total_frames = 18000
    bg_value = 10
    sigma = np.array([1.5, 1.5])

    _bg = lambda img: noise_psf(img, bg_poisson=bg_value)
    _psf = lambda pos, phot: gaussian_expect(pos, sigma, phot, img_shape=image_size)
    _psf_hr = lambda pos, phot: delta_psf(pos, phot, img_shape=image_size_hr,
                                          xextent=np.array([0, image_size[0]], dtype=float),
                                          yextent=np.array([0, image_size[1]], dtype=float))

    emit, cont = random_emitters(emitter_p_frame, total_frames, None, image_size, 3)
    sim = Simulation(emit, cont, img_size=image_size, upscale=upscale_factor,
                     background=_bg, psf=_psf, psf_hr=_psf_hr)
    sim.image = upscale(sim.camera_image(upscale=1), sim.upscale)  # don't upscale before convolution, scale final image
    sim.image_hr = sim.camera_image(psf=sim.psf_hr, bg=False, upscale=upscale_factor)  # work on hr, no need to upscale
    sim.write_to_binary(binary_path)

    plt.figure(figsize=(10, 10))
    plt.subplot(121)
    sim.plot_frame(0)
    plt.subplot(122)
    sim.plot_frame(0, image=sim.image_hr, crosses=False)
    plt.show()

    print("Generating samples done. Filename: {}".format(binary_path))
