def outsize_calc(i, p, k, s, d):
    """
    i = input_size
    o = output
    p = padding
    k = kernel_size
    s = stride
    d = dilation
    :return:
    """

    o = (i + 2 * p - k - (k - 1) * (d - 1)) / s + 1
    return o


def pad_same_calc(i, k, s, d):
    p = (s * (i -1) - i + k + (k - 1)*(d - 1)) / 2
    if p % 1 != 0:
        raise ValueError('Padding Same not possible.')

    return int(p)