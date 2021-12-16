import os
import re
from typing import *

import numpy as np
from kornia.geometry.subpix import spatial_soft_argmax2d
from torch.functional import Tensor

from dataset import Drosophila2Dataset
from model import Drosophila2DPose

import torch


def inference(model: Drosophila2DPose, dataset: Drosophila2Dataset) -> np.ndarray:
    res = list()
    for batch in dataset:
        x, _, d = batch
        hm = model(x)
        points = heatmap2points(hm)
        points = points.cpu().data.numpy()
        for idx in range(x.size(0)):
            path = d[0][idx]
            res.append([path, points[idx]])

    # print(res)
    # return None
    points2d = inp2np(res)

    return points2d


def parse_img_path(name: str) -> Tuple[int, int]:
    """returns cid and img_id """
    name = os.path.basename(name)
    match = re.match(r"camera_(\d+)_img_(\d+)", name.replace(".jpg", ""))
    return int(match[1]), int(match[2])


def inp2np(inp: List) -> np.ndarray:
    n_cameras = max([parse_img_path(p)[0] for (p, _) in inp]) + 1
    n_images = max([parse_img_path(p)[1] for (p, _) in inp]) + 1
    n_joints = inp[0][1].shape[0]

    points2d = np.ones((n_cameras, n_images + 1, n_joints, 2))

    for (path, pts) in inp:
        cid, imgid = parse_img_path(path)
        points2d[cid, imgid] = pts

    return points2d


def heatmap2points(x: Tensor) -> Tensor:
    """ B x C x H x W -> B x C x 2"""
    return spatial_soft_argmax2d(
        x, temperature=torch.tensor(10000), normalized_coordinates=True
    )


def path2inp(path: str):
    return [
        (path + p, np.zeros((19)))
        for p in os.listdir(path)
        if p.endswith(".jpg") or p.endswith(".png")
    ]

