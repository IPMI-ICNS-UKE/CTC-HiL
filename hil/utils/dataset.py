from typing import Iterable
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


def normalize(img: torch.Tensor, min_value: float, max_value: float) -> torch.Tensor:
    """
    Min max intensity normalization of torch image.

    Params:
        img (torch.Tensor):
        min_value (float): Minimum value of image.
        max_value (float): Maximum value of image.

    Returns:
        torch.Tensor: The normalized image tensor.

    """
    img = img.clamp(min=min_value, max=max_value)
    img = img - min_value
    img = img / (max_value - min_value)
    img = (img * 2) - 1
    return img


class Dataset_labeled(Dataset):
    def __init__(
            self,
            root_path,
            img_paths,
            label_list: Iterable = None,
            normalize_ctc=True,
    ):
        self.root_path = root_path
        self.img_paths = img_paths
        self.labels = label_list
        self.normalize_ctc = normalize_ctc

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        dapi_path = Path(self.root_path + self.img_paths[index][0])
        ck_path = Path(self.root_path + self.img_paths[index][1])
        cd45_path = Path(self.root_path + self.img_paths[index][2])

        """ Resize images to 224X224 """
        dapi_img = cv2.resize(np.array(Image.open(dapi_path).convert('L')), dsize=(224, 224),
                              interpolation=cv2.INTER_LINEAR)
        ck_img = cv2.resize(np.array(Image.open(ck_path).convert('L')), dsize=(224, 224),
                            interpolation=cv2.INTER_LINEAR)
        cd45_img = cv2.resize(np.array(Image.open(cd45_path).convert('L')), dsize=(224, 224),
                              interpolation=cv2.INTER_LINEAR)
        img = np.stack((dapi_img, ck_img, cd45_img))

        if self.normalize_ctc:
            img = torch.tensor(img) / 255
            img = normalize(img, 0, 1)

        label = self.labels[index]

        return img, label
