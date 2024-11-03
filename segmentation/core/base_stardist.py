import time

from typing import List

from csbdeep.utils import normalize
from alive_progress import alive_bar


class StarDistBase:
    """
    Base class for StarDist segmentation operations.
    """
    def __init__(self, model, logger=None):
        """
        Initialize the StarDist model with pretrained 2D_versatile_fluo model.

        Params:
            model: The name of the pretrained StarDist model to use.
        """
        self.model = model
        self.logger = logger

    def give_coords(self, ck_image_lst:List) -> List[List[List[float]]]:
        """
        Perform segmentation to obtain coordinates for each image.

        Params:
            ck_image_lst (List): List of images (NumPy arrays) to be processed.

        Returns:
             List[List[List[float]]]: List of coordinates for each image. Structure: coords_lst is a list that contains
                                      multiple elements. Each element in this list is also a list. Each of these inner
                                      list elements is a list of two lists (y, x). Each of these two lists contains
                                      floats that represent coordinates.
        """
        print(f"Performing StarDist segmentation")
        if self.logger:
            self.logger.debug(f"Predict instances")
        counter = 0
        coords_lst = []
        with alive_bar(len(ck_image_lst), force_tty=True) as bar:
            for ck_image in ck_image_lst:
                time.sleep(.005)
                labels, output_specs = self.model.predict_instances(normalize(ck_image))
                coords_lst.append(output_specs["coord"].tolist())
                counter = counter + 1
                bar()
        return coords_lst
