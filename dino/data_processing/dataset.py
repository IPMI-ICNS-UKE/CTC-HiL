from __future__ import annotations
from pathlib import Path
from typing import Callable, Tuple, Sequence, List
from abc import ABC, abstractmethod
from functools import partial

import cv2
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_tensor
from PIL import ImageFile, Image
from PIL.Image import Image as ImageType


from sparsam.utils import min_max_normalize_tensor

ImageFile.LOAD_TRUNCATED_IMAGES = True


class BaseSet(ABC, Dataset):
    def __init__(self,
                 data_augmentation: Callable = None,
                 normalize: Callable | bool = True
                 ):
        self.data_augmentation = data_augmentation
        if normalize is True:
            normalize = partial(min_max_normalize_tensor, min_value=0, max_value=1)
        self.normalize = normalize

    def set_data_augmentation(self, data_augmentation: Callable):
        self.data_augmentation = data_augmentation

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        # Get image and corresponding label
        img, label = self._get_image_label_pair(index)
        if self.data_augmentation:
            img = self.data_augmentation(img)
        return img, label

    def _normalize(self, img: Tensor | ImageType):
        if not isinstance(img, Tensor):
            img = to_tensor(img)
        return img

    @abstractmethod
    def _get_image_label_pair(self, index: int) -> Tuple[
        Tensor | ImageType | List[Tensor | ImageType], Tensor | int | None]:
        """
        Params:
            index: Which datapoint from the dataset to get.

        Returns:
            img: the loaded and preprocessed, but UNNORMALIZED image
            label: if dataset is labeled -> returns the corresponding image label or dummy label/ None
        """
        pass


class DinoImageSet(BaseSet):
    def __init__(
            self,
            img_paths: Sequence[Path],
            labels: Sequence = None,
            img_size: int | Sequence[int] = None,
            data_augmentation: Callable = None,
            class_names: Sequence[str] = None,
            normalize: Callable | bool = True,
    ):
        super().__init__(data_augmentation=data_augmentation, normalize=normalize)
        self.img_paths = img_paths
        self.labels = labels
        if class_names:
            self.class_names = class_names
        elif labels:
            self.class_names = sorted(list(set(labels)))
        else:
            self.class_names = None
        if img_size and not isinstance(img_size, tuple):
            img_size = (img_size, img_size)
        self.img_size = img_size

    def __len__(self):
        return len(self.img_paths)

    def _get_image_label_pair(self, index: int) -> Tuple[torch.Tensor, int]:
        # Load images, resize them and convert to tensor
        paths = self.img_paths[index]
        dapi_img = cv2.resize(np.array(Image.open(paths[0])), dsize=(224, 224), interpolation=cv2.INTER_LINEAR)
        ck_img = cv2.resize(np.array(Image.open(paths[1])), dsize=(224, 224), interpolation=cv2.INTER_LINEAR)
        cd45_img = cv2.resize(np.array(Image.open(paths[2])), dsize=(224, 224), interpolation=cv2.INTER_LINEAR)

        img = np.stack((dapi_img, ck_img, cd45_img)).swapaxes(0, -1)
        img = Image.fromarray(img)

        return img, 0