from operator import itemgetter
from typing import Callable


class TransformSequence:
    """
    Simple class which calls forward method of all it's components sequentially.
    """

    def __init__(self, components, input_slice=None):
        """

        Args:
            components: components with forward method
            input_slice: list of lists which indicate what is the output to the i-th component; e.g. [[0, 1], [0]]
            means that the first component get's the 0th and 1st element which are input to this instances forward
            method, the 1st component will get the 0th output of the 0th component. Input slice is ignored when the
            potential input is not a tuple anyways

        """
        self.com = components
        self._input_slice = input_slice

        """Sanity"""
        if self._input_slice is not None:
            assert len(self._input_slice) == len(self), "Input slices must be the same number as components"

    @classmethod
    def parse(cls, components, param: dict, **kwargs):
        """
        If all components implemented a parse method, you can do it globally only once for the whole sequence.

        Args:
            components: component reference (unintialised) with forward method
            param (dict): parameters which are forwarded to the constructor of the components
            kwargs: arbitrary keyword arguments subject to this class constructor

        returns:
            TransformSequence or subclass of it
        """
        return cls([cpt.parse(param) for cpt in components], **kwargs)

    def __len__(self):
        """
        Returns the number of components
        """
        return self.com.__len__()

    def forward(self, *x):
        """
        Forwards the input data sequentially through all components

        Args:
            *x: arbitrary input data

        Returns:
            Any: Output of the last component

        """

        for i, com in enumerate(self.com):

            if isinstance(x, tuple):

                if self._input_slice is not None:
                    com_in = itemgetter(*self._input_slice[i])(x)  # get specific outputs as input for next com
                    if len(self._input_slice[i]) >= 2:
                        x = com.forward(*com_in)
                    else:
                        x = com.forward(com_in)
                else:
                    x = com.forward(*x)
            else:
                x = com.forward(x)

        return x


class ParallelTransformSequence(TransformSequence):
    """
    Simple processing class that forwards data through all of it's components parallelly (not in a hardware sense) and
    returns a list of the output or combines them if a merging function is specified. A merging function needs to
    accept a list as an argument.

    """

    def __init__(self, components, input_slice, merger=None):
        super().__init__(components=components, input_slice=input_slice)

        self.merger = merger

    def forward(self, *x):

        out_cache = [None] * len(self)
        for i, com in enumerate(self.com):
            if self._input_slice is not None:
                com_in = itemgetter(*self._input_slice[i])(x)
                if len(self._input_slice[i]) >= 2:  # unpack
                    out_cache[i] = com.forward(*com_in)
                else:
                    out_cache[i] = com.forward(com_in)
            else:
                out_cache[i] = com.forward(*x)

        if self.merger is not None:
            return self.merger(out_cache)
        else:
            return out_cache


def wrap_callable(func: Callable):
    """
    Wrapps a callable in a class to provide a forward method. This is mainly a helper to wrap arbitrary functions to
    fit into the transform sequence as above.

    Args:
        func:

    """

    return _TrafoWrapper(func=func)


class _TrafoWrapper:
    """
    Wrapps a callable. Useful because this way they can be element of a Transform Sequence.
    Only to be used in conjunction with wrap_callable function above.
    """

    def __init__(self, func: Callable):
        self._wrapped_callable = func

    def forward(self, *args, **kwargs):
        return self._wrapped_callable(*args, **kwargs)
