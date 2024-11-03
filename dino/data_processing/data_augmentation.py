from __future__ import annotations
from abc import ABC
import random
# from typing import Callable, List, Sequence, Tuple, Union

from PIL import ImageOps
from PIL.Image import Image as ImageType
import numpy as np
import torch
from torch import Tensor
from torchvision.transforms import transforms, ToTensor
# from torchvision.transforms.functional import to_pil_image

from sparsam.utils import min_max_normalize_tensor


class BaseMultiCropper(ABC):
    def __init__(self, n_global_crops: int, n_local_crops: int):
        self.n_crops = n_global_crops + n_local_crops
        self.n_global_crops = n_global_crops
        self.n_local_crops = n_local_crops


class DataCropperDINO(BaseMultiCropper):
    """
    DataCropper for DINO.
    Adopted from sparsam with couple changes in augmentation/ transformations.
    """
    def __init__(
            self,
            n_global_crops: int,
            n_local_crops: int,
            global_crops_scale,
            local_crops_scale,
            res=256
    ):
        super(DataCropperDINO, self).__init__(n_global_crops, n_local_crops)
        # In DataAugmentation in Dataset
        flip_and_color_jitter = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomApply(
                    [transforms.ColorJitter(brightness=0.1, contrast=0.4, saturation=0.1, hue=0.1)], p=0.8
                ),
                # transforms.RandomGrayscale(p=0.2),
            ]
        )

        # First global crop
        self.global_transfo1 = transforms.Compose(
            [
                transforms.RandomPerspective(distortion_scale=0.2),
                transforms.RandomRotation(180),  # degrees=(0, 40)
                transforms.RandomInvert(p=0.3),
                transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.5),
                transforms.RandomAutocontrast(p=0.5),
                transforms.RandomEqualize(p=0.2),  #
                transforms.RandomResizedCrop(res, scale=global_crops_scale),
                flip_and_color_jitter,
                GaussianBlur(p=1.0, radius_min=0.1, radius_max=5.0),
                ToTensor(),
            ]
        )
        # Second global crop
        self.global_transfo2 = transforms.Compose(
            [
                transforms.RandomPerspective(distortion_scale=0.2),
                transforms.RandomRotation(180),
                transforms.RandomInvert(p=0.3),
                transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.5),
                transforms.RandomAutocontrast(p=0.5),
                transforms.RandomEqualize(p=0.2),  #
                transforms.RandomResizedCrop(res, scale=global_crops_scale),
                flip_and_color_jitter,
                GaussianBlur(p=0.1, radius_min=0.1, radius_max=5.0),
                Solarization(0.2),
                ToTensor(),
            ]
        )
        # Transformation for the local small crops
        self.local_transfo = transforms.Compose(
            [
                transforms.RandomPerspective(distortion_scale=0.2),
                transforms.RandomRotation(180),
                transforms.RandomInvert(p=0.3),  #
                transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.5),
                transforms.RandomAutocontrast(p=0.5),
                transforms.RandomEqualize(p=0.2),  #
                transforms.RandomResizedCrop(res // 3, scale=local_crops_scale),
                flip_and_color_jitter,
                GaussianBlur(p=0.5, radius_min=0.1, radius_max=5.0),
                ToTensor(),
            ]
        )

    def __call__(self, image: ImageType | Tensor | np.ndarray) -> ImageType:
        crops = []
        crops.append(min_max_normalize_tensor(self.global_transfo1(image), 0, 1))
        crops.append(min_max_normalize_tensor(self.global_transfo2(image), 0, 1))
        for _ in range(self.n_local_crops):
            crop = min_max_normalize_tensor(self.local_transfo(image), 0, 1)
            while torch.any(crop.isnan()):
                crop = torch.nan_to_num(crop, nan=0)
                print("NaN")

            crops.append(crop)
        return crops


class GaussianBlur(object):
    """
    Apply Gaussian Blur to the PIL image.
    """

    def __init__(self, p=0.5, radius_min=0.1, radius_max=2.):
        self.prob = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img):
        do_it = random.random() <= self.prob
        if not do_it:
            return img
        # TODO Lösung für Tensor
        blur_transforms = transforms.GaussianBlur(kernel_size=11,
                                                  sigma=random.uniform(self.radius_min, self.radius_max))
        return blur_transforms(img)


class Solarization(object):
    """
    Apply Solarization to the PIL image.
    """

    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return ImageOps.solarize(img)
        else:
            return img
