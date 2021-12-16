from typing import *
import torch
import matplotlib.pyplot as plt
import numpy as np
from skimage.transform import resize
from skimage.color import gray2rgb

# TODO implement augmentation
# TODO return also heatmaps


class Drosophila2Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        inp: List[List[Union[str, np.ndarray]]],
        img_size: List[int] = [256, 512],
        heatmap_size: List[int] = [64, 128],
        augmentation: bool = False,
    ):
        self.inp = inp
        self.img_size = img_size
        self.heatmap_size = heatmap_size
        self.augmentation = augmentation

    def __getitem__(self, idx):
        img_path, pts2d = self.inp[idx]
        img = plt.imread(img_path)
        img = resize(img, self.img_size)

        # gray2rgb
        if img.ndim == 2:
            img = gray2rgb(img)

        # h w c -> c h w
        img = np.swapaxes(img, 0, 2)

        return img, pts2d, tuple(self.inp[idx])

    def __len__(self):
        return len(self.inp)
