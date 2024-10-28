import os
import pickle
import re
from typing import *

from df2d.util import draw_labelmap
from matplotlib import pyplot as plt
import numpy as np
import torch.utils.data
import torchvision
import torchvision.transforms.v2 as v2
from skimage.transform import resize
from skimage.color import gray2rgb

def parse_img_path(name: str) -> Tuple[int, int]:
    """returns camera id and image id from image path like camera_3_img_234.jpg"""
    name = os.path.basename(name)
    match = re.match(r"camera_(\d+)_img_(\d+).(jpg|png)", name)
    if match is None:
        raise ValueError(f'Cannot parse image {name}')
    return int(match[1]), int(match[2])

def points_2d_to_heatmap(points2d: np.ndarray, heatmap_channels: int, heatmap_size: Tuple[int,int]):
    heatmap = np.zeros(
        (heatmap_channels, heatmap_size[0], heatmap_size[1])
    )
    present_joints = [
        idx
        for idx in range(heatmap_channels)
        if not np.all(points2d == 0)
    ]
    for idx in present_joints:
        heatmap[idx] = draw_labelmap(
            heatmap[idx], points2d * np.array(heatmap_size)
        )
    return heatmap

class Drosophila2Dataset(torch.utils.data.Dataset):
    """
    We provide the dataset as a list of image paths, which are scanned to find all the images.

    When getting an item from the dataset, the image at a particular path is loaded.

    Args:
        images_folder (str): Folder containing the images
        annotations_file (str): Pickle file containing the 2D points annotations for training
        camera_ids_to_flip (list[int]): flip images from these cameras so the fly is facing to the right
        max_num_imgs (int): the maximum number of images to process - eg. max_num_imgs=100 will process images 0 to 99
        img_size (tuple[int,int]): height and width to resize images before inputting them to the model
    """
    def __init__(
        self,
        images_folder: str,
        annotations_file: Optional[str],
        camera_ids_to_flip: List[int],
        max_num_imgs: Optional[int] = None,
        img_size: Optional[Tuple[int,int]] = (256, 512),
        heatmap_size: Tuple[int,int] = (64, 128),
        heatmap_channels: int = 19,
    ):
        self.images_folder=images_folder,
        self.camera_ids_to_flip = camera_ids_to_flip
        # self.transform = v2.Compose([
        #     v2.ToImage(),
        #     v2.ToDtype(torch.uint8, scale=True),
        #     v2.Resize(img_size),
        #     v2.ToDtype(torch.float32, scale=True), # Normalize expects float input
        #     v2.Normalize(mean=[0.22]*3, std=[1]*3), # I guess the model was trained with this normalisation?
        # ])
        self.img_size = img_size
        self.heatmap_size = heatmap_size
        self.heatmap_channels = heatmap_channels
        if annotations_file is not None:
            self.target_points2d = pickle.load(open(annotations_file, "rb"))["points2d"]
        else:
            self.target_points2d = None
        
        filename_pattern = re.compile(r'camera_\d*_img_\d*.(jpg|png)')
        self.imgs = [
            os.path.join(images_folder, path)
            for path in os.listdir(images_folder)
            if filename_pattern.match(path) and (max_num_imgs is None or parse_img_path(path)[1] < max_num_imgs)
        ]
        self.camera_indices = [parse_img_path(path)[0] for path in self.imgs]
        self.frame_indices = [parse_img_path(path)[1] for path in self.imgs]
        self.num_cameras = max(self.camera_indices) + 1
        self.num_images = max(self.frame_indices) + 1

        if self.num_cameras * self.num_images != len(self.imgs):
            raise ValueError(f"Can't find all images in dataset - found {self.num_cameras} cameras with {self.num_images} each, but only found {len(self.imgs)} total images")
        
    
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, tuple[int,int]]:
        """
        Args:
            index (int): Index of the data to get

        Returns:
            tuple: (sample, target, (camera_id, frame_id))
            - sample is the input image
            - target is the set of target 2D points
            - camera_id and frame_id are the indices of the image's camera and frame respectively
        """

        path = self.imgs[index]
        camera_id = self.camera_indices[index]
        frame_id = self.frame_indices[index]
        
        # image = torchvision.io.decode_image(path)
        # if camera_id in self.camera_ids_to_flip:
        #     image = v2.functional.horizontal_flip(image)
        # if self.transform is not None:
        #     image = self.transform(image)
        # loading model annotations for training isn't supported yet
        # if self.target_transform is not None:
        #     target = self.target_transform(target)
        
        image = plt.imread(path)
        if camera_id in self.camera_ids_to_flip:
            image = image[:,::-1]
        image = resize(image, self.img_size)
        # gray2rgb
        if image.ndim == 2:
            image = gray2rgb(image)
        # h w c -> c h w
        image = torch.Tensor(image.transpose(2, 0, 1))
        # remove the mean
        image = image - 0.22

        if self.target_points2d is not None:
            target = torch.Tensor(points_2d_to_heatmap(self.target_points2d[index], self.heatmap_channels, self.heatmap_size))
        else:
            target = torch.Tensor()

        return image, target, (camera_id, frame_id)
    
    def __len__(self):
        return len(self.imgs)

