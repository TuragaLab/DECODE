

class TransformSequence:
    """Simple class which calls forward method of all it's components."""
    def __init__(self, components):
        """

        :param components: instances with .forward method.
        input of k-th component must match output of (k-1)th component.
        """
        self.com = components

    def forward(self, x):
        """

        :param x: input
        :return: output
        """
        for com in self.com:
            x = com.forward(x)
        return x