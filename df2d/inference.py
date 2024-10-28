import os
import pickle
import re
import subprocess
from itertools import product
from typing import *

from df2d.parser import create_parser
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from df2d.dataset import Drosophila2Dataset
from df2d.model import Drosophila2DPose
from df2d.util import heatmap2points, pwd

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def download_weights(path):
    command = f"curl -L -o {path} https://www.dropbox.com/s/csgon8uojr3gdd9/sh8_front_j8.tar?dl=0"
    print("Downloading network weights.")
    os.makedirs(os.path.dirname(path))
    subprocess.run(command, shell=True)

# type annotations because the return of this function depends on the return_* bool args
@overload
def inference_folder(folder: str, camera_ids_to_flip: List[int], return_heatmap: Literal[False], return_confidence: Literal[False], max_img_id: Optional[int] = None, batch_size: int = 8, disable_pin_memory: bool = False, model_args: Dict[str,Any] = {}) -> np.ndarray: ...
@overload
def inference_folder(folder: str, camera_ids_to_flip: List[int], return_heatmap: Literal[True], return_confidence: Literal[False], max_img_id: Optional[int] = None, batch_size: int = 8, disable_pin_memory: bool = False, model_args: Dict[str,Any] = {}) -> Tuple[np.ndarray, np.ndarray]: ...
@overload
def inference_folder(folder: str, camera_ids_to_flip: List[int], return_heatmap: Literal[False], return_confidence: Literal[True], max_img_id: Optional[int] = None, batch_size: int = 8, disable_pin_memory: bool = False, model_args: Dict[str,Any] = {}) -> Tuple[np.ndarray, np.ndarray]: ...
@overload
def inference_folder(folder: str, camera_ids_to_flip: List[int], return_heatmap: Literal[True], return_confidence: Literal[True], max_img_id: Optional[int] = None, batch_size: int = 8, disable_pin_memory: bool = False, model_args: Dict[str,Any] = {}) -> Tuple[np.ndarray, np.ndarray,np.ndarray]: ...

def inference_folder(
    folder: str,
    camera_ids_to_flip: List[int],
    return_heatmap: bool = False,
    return_confidence: bool = False,
    max_img_id: Optional[int] = None,
    batch_size: int = 8, 
    disable_pin_memory: bool = False, 
    model_args: Dict[str,Any] = {},
):
    """processes all the images under a folder.
        returns normalized coordinates in [0, 1].
    >>> from df2d.inference import inference_folder
    >>> points2d = inference_folder('/home/user/Desktop/DeepFly3D/data/test/')
    >>> points2d.shape
    >>>     (7, 15, 19, 2) # n_cameras, n_images, n_joints, 2
    """
    checkpoint_path = os.path.join(pwd(), "../weights/sh8_deepfly.tar")
    if not os.path.exists(checkpoint_path):
        download_weights(checkpoint_path)
    args = create_parser().parse_args("").__dict__
    args.update(model_args)

    model = Drosophila2DPose(checkpoint_path=checkpoint_path, **args).to(device)
    model.eval() # makes pose estimation more deterministic during inference #5

    # See #7, from https://github.com/pytorch/pytorch/blob/0ff6f7a04083d3fb7f4084cc16175d6cce6ff4b5/torch/utils/data/dataloader.py#L592-L621 
    num_workers = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count() or 4
    dataset = Drosophila2Dataset(folder, None, camera_ids_to_flip, max_img_id+1 if max_img_id is not None else None)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=not disable_pin_memory)

    return inference(
        model, dataloader, return_heatmap=return_heatmap, return_confidence=return_confidence
    )


def pr2inp(path: str) -> List[str]:
    """converts pose result file into inp format
    >>> pr2inp("/data/test/df3d/df3d_result.pkl")
    >>>     [("/data/test/0.jpg", [[0,0], [.5,.5]]), ("/data/test/1.jpg", [[0,0], [.5,.5]])]
    """
    img_list = list()
    points2d = pickle.load(open(path, "rb"))["points2d"]
    n_cameras, n_images = points2d.shape[0] - 1, points2d.shape[1] - 1

    for cid, imgid in product(range(n_cameras), range(n_images)):
        img_path = os.path.join(
            os.path.dirname(path),
            "../camera_{cid}_img_{imgid}.jpg".format(cid=cid, imgid=imgid),
        )

        # skip if pose is missing
        if np.all(points2d[cid, imgid] == 0):
            continue

        if points2d[cid, imgid, 0, 0] == 0:
            pts = points2d[cid, imgid][19:]
            pts[..., 1] = 1 - pts[..., 1]
        else:
            pts = points2d[cid, imgid][:19]

        img_list.append((img_path, pts))

    return img_list


def path2inp(path: str, max_img_id: Optional[int] = None):
    """
    >>> path2inp("/data/test/")
    >>>     ["/data/test/0.jpg", "/data/test/1.jpg"]
    """
    pattern = re.compile(r'camera_\d*_img_\d*.(jpg|png)')
    img_list = [
        (os.path.join(path, p), np.zeros((19)))
        for p in os.listdir(path)
        if pattern.match(p)
    ]

    if max_img_id is not None:
        img_list = [img for img in img_list if parse_img_path(img[0])[1] <= max_img_id]

    return img_list


@torch.inference_mode()
def inference(
    model: Drosophila2DPose,
    dataloader: DataLoader,
    return_heatmap: bool = False,
    return_confidence: bool = False,
):   
    camera_ids_output: List[int] = []
    frame_ids_output: List[int] = []
    points_output: List[np.ndarray] = []
    conf_output: List[np.ndarray] = []
    heatmaps: List[np.ndarray] = []

    for batch in tqdm(dataloader):
        x, _, (camera_ids, frame_ids) = batch
        heatmap_batch = model(x).cpu()
        points_batch, conf_batch = heatmap2points(heatmap_batch)

        camera_ids_output += list(camera_ids.numpy())
        frame_ids_output += list(frame_ids.numpy())
        points_output += [points for points in points_batch.numpy()]
        if return_confidence:
            conf_output += [conf for conf in conf_batch.numpy()]

        if return_heatmap:
            heatmaps += [heatmap for heatmap in heatmap_batch.numpy()]

    points2d = reshape_outputs(points_output, camera_ids_output, frame_ids_output)

    if not return_heatmap and not return_confidence:
        return points2d
    
    ret = (points2d,)
    if return_confidence:
        conf = reshape_outputs(conf_output, camera_ids_output, frame_ids_output)
        ret += (conf,)
    if return_heatmap:
        heatmap = reshape_outputs(heatmap_batch, camera_ids_output, frame_ids_output)
        ret += (heatmap,)
    return ret

def parse_img_path(name: str) -> Tuple[int, int]:
    """returns camera id and image id from image path like camera_3_img_234.jpg"""
    name = os.path.basename(name)
    match = re.match(r"camera_(\d+)_img_(\d+).(jpg|png)", name)
    if match is None:
        raise ValueError(f'Cannot parse image {name}')
    return int(match[1]), int(match[2])


def reshape_outputs(outputs: List[np.ndarray], camera_ids: List[int], frame_ids: List[int]) -> np.ndarray:
    """
    Converts a list representation (length num_cameras*num_frames) of np.arrays into a numpy array of shape [num_cameras, num_frames, *input_array.shape].
    The ordering of the elements in the list representation matches the ordering of the camera_ids and frame_ids lists.
    """
    num_cameras = max(camera_ids) + 1
    num_frames = max(frame_ids) + 1

    reshaped_outputs = np.zeros((num_cameras, num_frames, *outputs[0].shape))

    for output, camera_id, frame_id in zip(outputs, camera_ids, frame_ids):
        reshaped_outputs[camera_id, frame_id] = output

    return reshaped_outputs
