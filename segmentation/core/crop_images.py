import os
import time

import pandas as pd
from typing import Tuple, Union, List
import cv2
import numpy as np
from alive_progress import alive_bar


def resize_bbox(
        x_min: float,
        x_max: float,
        y_min: float,
        y_max: float,
        crop_image_size: float
) -> Tuple[float, float, float, float]:
    """
    Resize 2D bounding box to desired size.

    Params:
        x_min (float): Upper-left corner x coordinate of the bounding box.
        x_max (float): Lower-right corner x coordinate.
        y_min (float): Upper-left corner y coordinate of the bounding box.
        y_max (float): Lower-right corner y coordinate.
        crop_image_size (float): Desired size for width and height of the new bounding box.

    Returns:
        Tuple[float, float, float, float]: New bounding box coordinates as (x_min, y_min, x_max, y_max).
    """
    width_box = x_max - x_min
    height_box = y_max - y_min
    center_coord = (x_min + (width_box / 2), y_min + (height_box / 2))
    x_min = center_coord[0] - (crop_image_size / 2)
    y_min = center_coord[1] - (crop_image_size / 2)
    x_max = center_coord[0] + (crop_image_size / 2)
    y_max = center_coord[1] + (crop_image_size / 2)

    return x_min, y_min, x_max, y_max


def check_bbox_in_image_dim(
        x_min: float,
        y_min: float,
        x_max: float,
        y_max: float,
        crop_image_size: float,
        width: float,
        height: float
) -> Tuple[List[float], List[float], List[float], List[float]]:
    """
    Verify if a bounding box is within the cartridge image dimensions.

    Params:
        x_min (float): Upper-left corner x coordinate of the bounding box.
        y_min (float): Upper-left corner y coordinate of the bounding box.
        x_max (float): Lower-right corner x coordinate.
        y_max (float): Lower-right corner y coordinate.
        crop_image_size (float): Desired crop size.
        width (float): Width of cartridge image.
        height (float): Height of cartridge image.

    Returns:
        Tuple[List[float], List[float], List[float], List[float]]: Lists containing the adjusted coordinates for the
                                                                   bounding box
    """

    x_min_lst = []
    x_max_lst = []
    y_min_lst = []
    y_max_lst = []

    if x_max <= width:
        if x_min < 0:
            x_min_lst.append(0)
            x_max_lst.append(int(crop_image_size))
        if x_min >= 0:
            x_min_lst.append(x_min)
            x_max_lst.append(x_max)
    if x_max > width:
        x_max_lst.append(width)
        diff = x_max - width
        x_min_new = x_min - diff
        x_min_lst.append(x_min_new)

    if y_max <= height:
        if y_min < 0:
            y_min_lst.append(0)
            y_max_lst.append(int(crop_image_size))
        if y_min >= 0:
            y_min_lst.append(y_min)
            y_max_lst.append(y_max)

    if y_max > height:
        y_max_lst.append(height)
        diff = y_max - height
        y_min_new = y_min - diff
        y_min_lst.append(y_min_new)
    return x_min_lst, y_min_lst, x_max_lst, y_max_lst


def save_cropped_images(
        save_path: str,
        image_num: int,
        label: Union[int, str],
        dapi_channel: int,
        cropped_dapi_image: np.ndarray,
        ck_channel: int,
        cropped_ck_image: np.ndarray,
        cd45_channel: int,
        cropped_cd45_image: np.ndarray
) -> Tuple[str, str, str]:
    """
    Save the cropped images to the specified directory.

    Returns:
        Paths of the saved cropped DAPI, CK, and CD45 images.
    """
    base_directory = os.path.join(save_path, "cropped_images/all")
    cropped_dapi_image_path = os.path.join(base_directory, f"{image_num}_{dapi_channel}_{label}.png")
    cropped_ck_image_path = os.path.join(base_directory, f"{image_num}_{ck_channel}_{label}.png")
    cropped_cd45_image_path = os.path.join(base_directory, f"{image_num}_{cd45_channel}_{label}.png")

    cv2.imwrite(cropped_dapi_image_path, cropped_dapi_image)
    cv2.imwrite(cropped_ck_image_path, cropped_ck_image)
    cv2.imwrite(cropped_cd45_image_path, cropped_cd45_image)

    return cropped_dapi_image_path, cropped_ck_image_path, cropped_cd45_image_path


class BoundingBox(object):
    """
    Creates a 2D (smallest possible) bounding box based on coords/ points.
    """

    def __init__(self, points: List[Tuple[float, float]]):
        # points = List[Tuple[x,y]]
        if len(points) == 0:
            raise ValueError("Can't compute bounding box of empty list")
        self.x_min, self.y_min = float("inf"), float("inf")
        self.x_max, self.y_max = float("-inf"), float("-inf")
        for x, y in points:
            # Set min coords
            if x < self.x_min:
                self.x_min = x
            if y < self.y_min:
                self.y_min = y
            # Set max coords
            if x > self.x_max:
                self.x_max = x
            elif y > self.y_max:
                self.y_max = y

    @property
    def width(self) -> float:
        """Return the width of the bounding box."""
        return self.x_max - self.x_min

    @property
    def height(self) -> float:
        """Return the height of the bounding box."""
        return self.y_max - self.y_min

    def __repr__(self) -> str:
        return "BoundingBox({}, {}, {}, {})".format(
            self.x_min, self.x_max, self.y_min, self.y_max)


class Cropper:
    """
    Crop images based on bounding box coordinates.
    """

    def __init__(self, logger=None, label=int(10)):
        # Default label is 10 meaning unknown (1= CTC, 0=CTC)
        self.logger = logger
        self.label = label
        self.dataframe = pd.DataFrame({
            "case": pd.Series(dtype="str"),
            "frame_num": pd.Series(dtype="float"),
            "label": pd.Series(dtype="float"),
            "box_length": pd.Series(dtype="float"),
            "x_min": pd.Series(dtype="float"),
            "y_min": pd.Series(dtype="float"),
            "x_max": pd.Series(dtype="float"),
            "y_max": pd.Series(dtype="float"),
            "dapi_path": pd.Series(dtype="str"),
            "ck_path": pd.Series(dtype="str"),
            "cd45_path": pd.Series(dtype="str")

        })

    def crop_images(
            self,
            case: str,
            ck_channel: int,
            ck_image_lst: List[np.ndarray],
            ck_image_path_lst: List[str],
            dapi_channel: int,
            dapi_image_lst: List[np.ndarray],
            cd45_channel: int,
            cd45_image_lst: List[np.ndarray],
            segmentation_coords_lst: List[List[List[float]]],
            crop_image_size: float,
            width: float,
            height: float,
            save_path: str
    ) -> Tuple[pd.DataFrame, List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
        """
        Crop images from cartridge images based on stardist segmentation coordinates.

        Params:
            case (str): Cartridge number.
            ck_channel (int): CK channel index.
            ck_image_lst (List[np.ndarray]): List of CK images.
            ck_image_path_lst (List[str]): List of paths to CK images.
            dapi_channel (int): DAPI channel index.
            dapi_image_lst (List[np.ndarray]): List of DAPI images.
            cd45_channel (int): CD45 channel index.
            cd45_image_lst (List[np.ndarray]): List of CD45 images.
            segmentation_coords_lst (List[List[Tuple[float, float]]]): List of segmentation coordinates.
            crop_image_size (float): Size to which images are cropped.
            width (float): Width of the cartridge image.
            height (float): Height of the cartridge image.
            save_path (str): Path to save the cropped images.

        Returns:
             Tuple containing the updated dataframe and lists of cropped images.
        """
        counter = 0
        cropped_dapi_image_lst = []
        cropped_ck_image_lst = []
        cropped_cd45_image_lst = []
        coords_lst = []

        print(f"Cropping")
        with alive_bar(len(ck_image_lst), force_tty=True) as bar:
            for image_num in range(len(ck_image_lst)):
                time.sleep(.005)
                dapi_image = dapi_image_lst[image_num]
                ck_image = ck_image_lst[image_num]
                cd45_image = cd45_image_lst[image_num]

                # Extract frame number from path
                frame_num = int(ck_image_path_lst[image_num].split("/")[-1].split("p")[-1].split("-")[0])

                # Extract coordinates for the current image
                coords_of_current_img = segmentation_coords_lst[image_num]

                if len(coords_of_current_img) >= 0:
                    for coords in coords_of_current_img:
                        # Coords order: [[y coords], [x coords]]
                        x_coords = coords[1]
                        y_coords = coords[0]
                        coords_lst.append(list(zip(x_coords, y_coords)))

                        # Get the smallest possible bounding box
                        box_coords = BoundingBox(list(zip(x_coords, y_coords)))

                        self.dataframe.loc[counter, "case"] = case
                        self.dataframe.loc[counter, "frame_num"] = frame_num
                        self.dataframe.loc[counter, "label"] = self.label

                        # Resize bounding box to desired size
                        x_min_resized, y_min_resized, x_max_resized, y_max_resized = resize_bbox(
                            x_min=box_coords.x_min,
                            x_max=box_coords.x_max,
                            y_min=box_coords.y_min,
                            y_max=box_coords.y_max,
                            crop_image_size=crop_image_size
                        )

                        # Check if bounding box is within image dimensions
                        x_min_new, y_min_new, x_max_new, y_max_new = check_bbox_in_image_dim(
                            x_min=x_min_resized,
                            y_min=y_min_resized,
                            x_max=x_max_resized,
                            y_max=y_max_resized,
                            crop_image_size=crop_image_size,
                            width=width, height=height
                        )

                        self.dataframe.loc[counter, "x_min"] = x_min_new[0]
                        self.dataframe.loc[counter, "y_min"] = y_min_new[0]
                        self.dataframe.loc[counter, "x_max"] = x_max_new[0]
                        self.dataframe.loc[counter, "y_max"] = y_max_new[0]

                        # Crop images
                        cropped_dapi_image = dapi_image[
                                             int(y_min_new[0]):int(y_max_new[0]),
                                             int(x_min_new[0]):int(x_max_new[0])]
                        cropped_ck_image = ck_image[
                                           int(y_min_new[0]):int(y_max_new[0]),
                                           int(x_min_new[0]):int(x_max_new[0])]
                        cropped_cd45_image = cd45_image[
                                             int(y_min_new[0]):int(y_max_new[0]),
                                             int(x_min_new[0]):int(x_max_new[0])]

                        # Add cropped images to lists
                        cropped_dapi_image_lst.append(cropped_dapi_image)
                        cropped_ck_image_lst.append(cropped_ck_image)
                        cropped_cd45_image_lst.append(cropped_cd45_image)

                        # Save cropped images
                        cropped_dapi_image_path, cropped_ck_image_path, cropped_cd45_image_path = \
                            save_cropped_images(
                                save_path=save_path,
                                image_num=counter + 1,
                                label=self.label,
                                dapi_channel=dapi_channel,
                                cropped_dapi_image=cropped_dapi_image,
                                ck_channel=ck_channel,
                                cropped_ck_image=cropped_ck_image,
                                cd45_channel=cd45_channel,
                                cropped_cd45_image=cropped_cd45_image
                            )

                        # Record paths in the dataframe
                        self.dataframe.loc[counter, "dapi_path"] = cropped_dapi_image_path
                        self.dataframe.loc[counter, "ck_path"] = cropped_ck_image_path
                        self.dataframe.loc[counter, "cd45_path"] = cropped_cd45_image_path

                        # Record the size of the cropped image
                        box_length = np.shape(cropped_ck_image)
                        self.dataframe.loc[counter, "box_length"] = box_length[0]
                        counter = counter + 1
                bar()
        if self.logger:
            self.logger.info(f"Cropped {len(self.dataframe)} cells or objects.")
        self.dataframe.to_csv(os.path.join(save_path, "dataframe", f"{case}_segmentation_dataframe"), index=False)
        return self.dataframe, cropped_dapi_image_lst, cropped_ck_image_lst, cropped_cd45_image_lst
