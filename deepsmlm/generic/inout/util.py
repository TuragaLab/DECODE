from pathlib import Path


def add_root_relative(path: (str, Path), root: (str, Path)):
    """
    Adds the root to a path if the path is not absolute

    Args:
        path (str, Path): path to file
        root (str, Path): root path

    Returns:
        Path: absolute path to file

    """
    if not isinstance(path, Path):
        path = Path(path)

    if not isinstance(root, Path):
        root = Path(root)

    if path.is_absolute():
        return path

    else:
        return root / path