import os
from itertools import product
from typing import Tuple

import numpy as np
import torch
from torch.functional import Tensor
from torchvision import utils


def pwd():
    return os.path.dirname(os.path.realpath(__file__))


def heatmap2points(x: Tensor) -> Tuple[Tensor,Tensor]:
    """B x C x H x W -> B x C x 2"""
    out = torch.zeros((x.size(0), x.size(1), 2), device=x.device)
    conf = torch.zeros((x.size(0), x.size(1), 1), device=x.device)
    for b, c in product(range(x.size(0)), range(x.size(1))):
        conf[b, c] = torch.max(x[b, c])
        out[b, c] = (x[b, c] == conf[b, c]).nonzero(as_tuple=False)[0]
        out[b, c] /= torch.tensor([x.size(2), x.size(3)], device=x.device)
    return out, conf


def heatmap_peaks(heatmaps,
                  k: int = 10,
                  min_distance: int = 1,
                  threshold_rel: float = 0.5) -> np.ndarray:
    """
    Extract up to `k` local-maximum peaks per channel from a heatmap batch.

    Used to feed multiple 2D-pose candidates per joint into the pictorial-
    structures (belief propagation) pose-correction step.

    Parameters
    ----------
    heatmaps : torch.Tensor or np.ndarray
        Shape (B, C, H, W). Tensors are moved to CPU and converted to numpy.
    k : int
        Maximum number of peaks per channel.
    min_distance : int
        Minimum pixel separation between accepted peaks.
    threshold_rel : float
        Threshold as a fraction of the per-channel maximum. Pixels below
        this value are not considered as peak candidates.

    Returns
    -------
    peaks : np.ndarray
        Shape (B, C, k, 3), dtype float32. Last axis: (y_normalized,
        x_normalized, score). y and x are divided by H and W respectively,
        matching the convention of `heatmap2points`. Slots beyond the
        number of detected peaks are padded with (0, 0, 0); padded entries
        can be distinguished from real ones by `score == 0`.
    """
    from scipy.ndimage import maximum_filter

    if isinstance(heatmaps, Tensor):
        heatmaps = heatmaps.cpu().numpy()

    B, C, H, W = heatmaps.shape
    footprint_size = 2 * min_distance + 1
    out = np.zeros((B, C, k, 3), dtype=np.float32)
    for b in range(B):
        for c in range(C):
            hm = heatmaps[b, c]
            hm_max = hm.max()
            if hm_max <= 0:
                continue
            local_max = maximum_filter(hm, size=footprint_size) == hm
            above_threshold = hm > (hm_max * threshold_rel)
            ys, xs = np.nonzero(local_max & above_threshold)
            if ys.size == 0:
                continue
            scores = hm[ys, xs]
            order = np.argsort(-scores)[:k]
            n = order.size
            out[b, c, :n, 0] = ys[order] / H
            out[b, c, :n, 1] = xs[order] / W
            out[b, c, :n, 2] = scores[order]
    return out


# draw 2d gaussians
def draw_labelmap(img: np.ndarray, pt, sigma=3, type="Gaussian"):
    # Draw a 2D gaussian
    # Adopted from https://github.com/anewell/pose-hg-train/blob/master/src/pypose/draw.py
    pt[0], pt[1] = pt[1], pt[0]

    # Check that any part of the gaussian is in-bounds
    ul = [int(pt[0] - 3 * sigma), int(pt[1] - 3 * sigma)]
    br = [int(pt[0] + 3 * sigma + 1), int(pt[1] + 3 * sigma + 1)]

    # Generate gaussian
    size = 6 * sigma + 1
    x = np.arange(0, size, 1, float)
    y = x[:, np.newaxis]
    x0 = y0 = size // 2
    # The gaussian is not normalized, we want the center value to equal 1
    if type == "Gaussian":
        g = np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))

    # Usable gaussian range
    g_x = max(0, -ul[0]), min(br[0], img.shape[1]) - ul[0]
    g_y = max(0, -ul[1]), min(br[1], img.shape[0]) - ul[1]
    # Image range
    img_x = max(0, ul[0]), min(br[0], img.shape[1])
    img_y = max(0, ul[1]), min(br[1], img.shape[0])

    img[img_y[0] : img_y[1], img_x[0] : img_x[1]] = g[g_y[0] : g_y[1], g_x[0] : g_x[1]]
    return img


def tensorboard_plot_image(
    writer, x: Tensor, name: str, epoch: int, scale_each: bool = False
) -> None:
    """
    records single image into tensorboard.
    tensorboard logs are saved under lightning_logs.
    >>> SU.stensorboard_plot_image(
        self.logger.experiment,
        torch.cat([img1[:, :, -1], img2[:, :, -1]], dim=2),
        "recons",
        self.current_epoch,
    )
    """
    with torch.no_grad():
        x = x[:16].clone()
        grid = utils.make_grid(x, nrow=4, scale_each=scale_each, normalize=True)
        writer.add_image(name, grid, epoch)
