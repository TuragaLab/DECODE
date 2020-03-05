import warnings


class Simulation:
    """
    A simulation class that holds the necessary modules, i.e. an emitter source (either a static EmitterSet or
    a function from which we can sample emitters), a psf background and noise. You may also specify the desired frame
    range, i.e. the indices of the frames you want to have as output. If they are not specified, they are automatically
    determined but may vary with new sampled emittersets.

    Attributes:
        em (EmitterSet): Static EmitterSet
        em_sampler: callable to sample EmitterSets from
        frame_range (tuple of int): frame indices between which to sample
        psf (PSF): psf model with forward method
        background (Background): background implementation
        noise (Noise): noise implementation
    """

    def __init__(self, psf, em=None, em_sampler=None, background=None, noise=None, frame_range=None):
        """
        Init Simulation.

        Args:
            psf:
            em:
            em_sampler:
            background:
            noise:
            frame_range:
        """

        self.em = em
        self.em_sampler = em_sampler
        self.frame_range = frame_range if frame_range is not None else (None, None)

        self.psf = psf
        self.background = background
        self.noise = noise

        if (self.background is not None) and (self.noise is None):
            """This is temporary since I changed the interface."""
            warnings.warn("Careful! You have not specified noise but you have specified background. "
                          "Background is defined as something which does not depend on the actual "
                          "signal whereas noise does.")

    def forward(self, em_new=None):

        if em_new is not None:
            self.em = em_new
        elif self.em_sampler is not None:
            self.em = self.em_sampler()

        frames = self.psf.forward(self.em.xyz_px, self.em.phot, self.em.frame_ix,
                                  ix_low=self.frame_range[0], ix_high=self.frame_range[1])

        """
        Add background. This needs to happen here and not on a single frame, since background may be correlated.
        The difference between background and noise is, that background is assumed to be independent of the 
        emitter position / signal.
        """
        if self.background is not None:
            frames, bg_frames = self.background.forward(frames)
        else:
            bg_frames = None

        if self.noise is not None:
            frames = self.noise.forward(frames)

        return frames, bg_frames, self.em
