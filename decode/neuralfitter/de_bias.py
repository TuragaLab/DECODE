import numpy as np
import torch


class UniformizeOffset:
    """
    Rescales x and y offsets (inplace) so that they are distributed uniformly within [-0.5, 0.5] to correct for biased outputs.
    """

    def __init__(self, n_bins: int):
        """
        Args:
            n_bins (int): The bias scales with the uncertainty of the localization. Therefore all detections are binned according to their predicted uncertainty.
            Detections within different bins are then rescaled seperately. This specifies the number of bins.
        """
        self.n_bins = n_bins

    def histedges_equal_n(self, x):
        npt = len(x)
        return np.interp(np.linspace(0, npt, self.n_bins + 1),
                         np.arange(npt),
                         np.sort(x))

    def uniformize(self, x):
        x = np.clip(x, -0.99, 0.99)
        x_cdf = np.histogram(x, bins=np.linspace(-1, 1, 201))
        x_re = self.cdf_get(np.cumsum(x_cdf[0]) / sum(x_cdf[0]), x)
        return (x_re - 0.5).astype('float32')

    def cdf_get(self, cdf, val):
        ind = (val + 1) / 2 * 200 - 1.
        dec = ind - np.floor(ind)
        return (dec * cdf[[int(i) + 1 for i in ind]] + (1 - dec) * cdf[[int(i) for i in ind]])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Rescales x and y offsets (inplace) so that they are distributed uniformly within [-0.5, 0.5] to correct for biased outputs.
        Forward frames through post-processor.
        Args:
            x (torch.Tensor): features to be converted. Expecting x/y coordinates in channel index 2, 3 and x/y sigma coordinates in channel index 6, 7
             expected shape :math:`(N, C, H, W)`
        """
        device = x.device
        x = x.cpu().numpy()

        active_px = (x[:, 0] > 0.3).nonzero()
        x_sigma = x[:, 6]
        y_sigma = x[:, 7]
        x_sigma_var = np.var(x_sigma[active_px])
        y_sigma_var = np.var(y_sigma[active_px])
        weighted_sig = x_sigma ** 2 + (np.sqrt(x_sigma_var / y_sigma_var) * y_sigma) ** 2
        weighted_sig = np.where(x[:, 0] > 0.3, weighted_sig, 0)
        bins = self.histedges_equal_n(weighted_sig[active_px])
        for i in range(self.n_bins):
            inds = np.where((weighted_sig > bins[i]) & (weighted_sig < bins[i + 1]) & (weighted_sig != 0))
            x[:, 2][inds] = self.uniformize(x[:, 2][inds]) + np.mean(x[:, 2][inds])
            x[:, 3][inds] = self.uniformize(x[:, 3][inds]) + np.mean(x[:, 3][inds])

        return torch.tensor(x).to(device)
