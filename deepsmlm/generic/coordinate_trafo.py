import torch


class UpsamplingTransformation:
    """
    Simple class to calculate transformation from
    uspampled coordinates to original extent or vice versa
    """
    def __init__(self, extent, up_factor=None):
        """

        :param extent: extent of image
        :param up_factor: upsampling factor
        """
        self.extent = extent
        self.up_factor = up_factor

        """Calculate extent of upsampled image in 1D """
        up_pseudo_extent_1d = (-0.5, (extent[0][1] - extent[0][0]) * self.up_factor - 0.5)
        self.lin_m = self.up_factor
        self.lin_b = up_pseudo_extent_1d[1] - self.lin_m * extent[0][1]

        self.lin_mi = 1 / self.lin_m
        self.lin_bi = extent[0][1] - self.lin_mi * up_pseudo_extent_1d[1]

    def up2coord(self, coord_up):
        return self.lin_mi * coord_up.type(torch.FloatTensor) + self.lin_bi

    def coord2up(self, coord):
        return self.lin_m * coord.type(torch.FloatTensor) + self.lin_b


class A2BTransform:
    """Simple class to transform a frame to one coordinate system to another."""
    def __init__(self, a_extent, b_extent):
        """

        :param a_extent: extent in 2D of frame A
        :param b_extent: extent in 2D of frame B
        """
        if not (a_extent[0] == a_extent[1]) or not (b_extent[0] == b_extent[1]):
            raise ValueError("Matrices are not square.")

        self.a_extent = a_extent[0]
        self.b_extent = b_extent[0]

        self.ab_scale = (self.b_extent[1] - self.b_extent[0]) / (self.a_extent[1] - self.a_extent[0])
        self.a_shift = self.a_extent[0]
        self.b_shift = self.b_extent[0]

    @staticmethod
    def transform(x, x_shift, y_shift, scale):
        return (x - x_shift) * scale + y_shift

    def a2b(self, a):
        """
        Transform from a to b. Only square matrices.

        :param a: coordinates in frame a
        :return: coordinates in frame b
        """
        if a.dtype not in (torch.float, torch.double, torch.half):
            raise ValueError("Wrong datatype. Must be floating point tensor.")
        return self.transform(a, self.a_shift, self.b_shift, self.ab_scale)

    def b2a(self, b):
        """
        Transform from b to a. Onyl square matrices.

        :param b: coordinates in frame b
        :return: coordinates in frame a
        """
        if b.dtype not in (torch.float, torch.double, torch.half):
            raise ValueError("Wrong datatype. Must be floating point tensor.")
        return self.transform(b, self.b_shift, self.a_shift, 1 / self.ab_scale)


if __name__ == '__main__':
    f = torch.tensor([[0., 0]])
    print(A2BTransform(a_extent=((-0.5, 63.5), (-0.5, 63.5)), b_extent=((-0.5, 511.5), (-0.5, 511.5))).a2b(f))
