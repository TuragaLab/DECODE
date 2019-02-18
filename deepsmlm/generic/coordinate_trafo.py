import torch


class UpsamplingTransformation:
    """
    Simple class to calculate transformation from
    uspampled coordinates to original extent or vice versa
    """
    def __init__(self, extent, up_factor):
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
