import os
from functools import partial
import time
from pathlib import Path
import PIL.Image as Image
import numpy as np
import cv2
from typing import Union, List
import pandas as pd
from natsort import natsorted
from alive_progress import alive_bar

from segmentation.core.base_stardist import StarDistBase
from segmentation.core.crop_images import Cropper


class SegProcessorLogic:
    """
    The SegProcessorLogic class handles multi-frame TIFF images,
    splits them into separate single-channel images,
    performs the segmentation with the StarDist algorithm,
    and stores the processed results (inlcuding the cropped images).
    """

    def __init__(
            self,
            case: str,
            cartridge_path: str,
            save_path: str,
            model,
            img_size: Union[int, float],
            cartridge_width: int = 1384,
            cartridge_height: int = 1036,
            channel_num: int = 4,
            dapi_channel: int = 1,
            ck_channel: int = 3,
            cd45_channel: int = 4,
            logger=None
    ):
        self.case = case
        self.cartridge_path = cartridge_path
        self.save_path = save_path
        self.model = model
        self.crop_image_size = img_size

        self.logger = logger

        self.cartridge_width = cartridge_width
        self.cartridge_height = cartridge_height

        # Channel order of input images (cartridge images)
        self.channel_num = channel_num
        self.dapi_channel = dapi_channel
        self.ck_channel = ck_channel
        self.cd45_channel = cd45_channel

        # Paths for cartridge images stored as multi-frame TIFs
        self.cartridge_image_path_lst = []
        # Paths for single-channel TIF cartridge images
        self.cartridge_dapi_image_path_lst = []
        self.cartridge_ck_image_path_lst = []
        self.cartridge_cd45_image_path_lst = []
        # Numpy arrays for single-channel cartridge images
        self.cartridge_dapi_image_lst = []
        self.cartridge_ck_image_lst = []
        self.cartridge_cd45_image_lst = []

        # Stardist segmentation
        self.segmentation_coords_lst = []
        self.segmentation = StarDistBase(model=self.model, logger=self.logger)

        # Cropped images
        self.cropped_dapi_image_lst = []
        self.cropped_ck_image_lst = []
        self.cropped_cd45_image_lst = []

        self.crop_images = Cropper(logger=self.logger)

        self.segmentation_dataframe = pd.DataFrame()

    def get_cartridge_images(self) -> List[str]:
        """Collects and sorts all TIFF image paths from the input directory."""
        cartrdige_image_path_lst = []
        for image_path in os.listdir(self.cartridge_path):
            path_name = os.path.join(self.cartridge_path, image_path)
            path_name = Path(path_name).as_posix()  # for windows
            if path_name.lower().endswith(('.tif', '.tiff')):
                cartrdige_image_path_lst.append(path_name)
        return natsorted(cartrdige_image_path_lst)

    def split_multiframe_tif(self):
        """
        Splits multi-frame TIF files into single-channel images and saves them.
        """
        if self.logger:
            self.logger.info(f"Splitting {len(self.cartridge_image_path_lst)} TIFF files")
        with alive_bar(len(self.cartridge_image_path_lst), force_tty=True) as bar:
            for image_path in self.cartridge_image_path_lst:
                time.sleep(.005)
                image = Image.open(image_path)
                if image_path.split("/")[-1].startswith("cartid"):
                    image_name = "cartid" + image_path.split("cartid")[-1].split(".tif")[0] + "-000"
                else:
                    image_name = image_path.split("/")[-1].split(".tif")[0] + "-000"
                image_path_name = os.path.join(self.save_path, "cartridge_images",  image_name)
                for image_frame in range(self.channel_num):
                    image.seek(image_frame)
                    if image_frame == 0:
                        self.cartridge_dapi_image_lst.append(np.array(image))
                        current_image_path = image_path_name + str(self.dapi_channel) + ".tif"
                        self.cartridge_dapi_image_path_lst.append(current_image_path)
                        cv2.imwrite(current_image_path, np.array(image))
                    if image_frame == 2:
                        self.cartridge_ck_image_lst.append(np.array(image))
                        current_image_path = image_path_name + str(self.ck_channel) + ".tif"
                        self.cartridge_ck_image_path_lst.append(current_image_path)
                        cv2.imwrite(current_image_path, np.array(image))
                    if image_frame == 3:
                        self.cartridge_cd45_image_lst.append(np.array(image))
                        current_image_path = image_path_name + str(self.cd45_channel) + ".tif"
                        self.cartridge_cd45_image_path_lst.append(current_image_path)
                        cv2.imwrite(current_image_path, np.array(image))
                bar()

    def create_folders(self):
        """Creates output directories for images and data if they do not already exist."""
        concat_path = partial(os.path.join, self.save_path)
        subfolders = ('cartridge_images', 'dataframe', 'cropped_images/all')
        for subfolder in map(concat_path, subfolders):
            os.makedirs(subfolder, exist_ok=True)

    def start_segmentation(self):
        """
        Initiates the segmentation processing pipeline:
        loads images, performs segmentation, and stores processed results.
        """
        if self.logger:
            self.logger.debug("Start of segmentation pipeline")
        self.create_folders()
        self.cartridge_image_path_lst = self.get_cartridge_images()
        self.split_multiframe_tif()
        self.segmentation_coords_lst = self.segmentation.give_coords(self.cartridge_ck_image_lst)
        self.segmentation_dataframe, self.cropped_dapi_image_lst, self.cropped_ck_image_lst, \
            self.cropped_cd45_image_lst = self.crop_images.crop_images(self.case,
                                                                       self.ck_channel,
                                                                       self.cartridge_ck_image_lst,
                                                                       self.cartridge_ck_image_path_lst,
                                                                       self.dapi_channel,
                                                                       self.cartridge_dapi_image_lst,
                                                                       self.cd45_channel,
                                                                       self.cartridge_cd45_image_lst,
                                                                       self.segmentation_coords_lst,
                                                                       self.crop_image_size,
                                                                       self.cartridge_width,
                                                                       self.cartridge_height,
                                                                       self.save_path)
