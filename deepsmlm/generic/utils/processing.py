

class TransformSequence:
    """Simple class which calls forward method of all it's components."""
    def __init__(self, components):
        """

        :param components: instances with .forward method.
        input of k-th component must match output of (k-1)th component.
        """
        self.com = components

    @staticmethod
    def parse(components, param: dict):
        """
        If all components implemented a parse method, you can do it globally only once for the whole sequence.
        :param components: as in init.
        :param param:
        :return:
        """
        return TransformSequence([cpt.parse(param) for cpt in components])

    def forward(self, *x):
        """

        :param x: input
        :return: output
        """
        for com in self.com:
            if isinstance(x, tuple):
                x = com.forward(*x)
            else:
                x = com.forward(x)
        return x