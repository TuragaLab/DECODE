def add_root_relative(path, root):
    """
    Add root if path is relative
    :param path:
    :param root:
    :return:
    """
    if (path is not None) and (path[0] not in (None, "", "/")):
        return root + path
    else:
        return path