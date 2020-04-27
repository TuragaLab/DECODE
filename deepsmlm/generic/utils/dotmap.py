import dotmap


def convert(convertible, convert_type=dotmap.DotMap):
    """
    A nice little recursive algorithm for converting a type that is convertible to a dict to dict.

    Args:
        param
    """
    con_dict = dict(convertible)

    for k, v in con_dict.items():
        if isinstance(v, convert_type):
            con_dict[k] = convert(v, convert_type)

    return con_dict


class DotMap(dotmap.DotMap):
    """Extension of the DotMap class"""

    def convert_dict(self) -> dict:
        """
        Convert dotmap including all attributes to dictionary, i.e. all sub-dotmaps are converted as well.

        Returns:
            dict
        """
        return convert(self, convert_type=type(self))
