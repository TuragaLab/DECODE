from operator import itemgetter


class TransformSequence:
    """Simple class which calls forward method of all it's components."""
    def __init__(self, components, input_slice=None):
        """

        Args:
            components: components with forward method
            input_slice: select which output elements of component i-1 are input to component i. Specification starts
            with index of output of the 0th element that will be input to the 1st com.
        """
        self.com = components
        self._input_slice = input_slice

        """Sanity"""
        if self._input_slice is not None:
            assert len(self._input_slice) == len(self) - 1, "Input slices must be one less than number of components"

    @staticmethod
    def parse(components, param: dict, **kwargs):
        """
        If all components implemented a parse method, you can do it globally only once for the whole sequence.
        :param components: as in init.
        :param param:
        :return:
        """
        return TransformSequence([cpt.parse(param) for cpt in components], **kwargs)

    def __len__(self):
        """
        Returns the number of components
        :return:
        """
        return self.com.__len__()

    def forward(self, *x):
        """

        :param x: input
        :return: output
        """
        for i, com in enumerate(self.com):
            if isinstance(x, tuple):
                if self._input_slice is not None and i >= 1:
                    com_in = itemgetter(*self._input_slice[i-1])(x)  # get specific outputs as input for next com
                    x = com.forward(com_in)
                else:
                    x = com.forward(*x)
            else:
                x = com.forward(x)
        return x