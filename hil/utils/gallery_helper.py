import os
import platform

from typing import Tuple, Optional
from collections import Counter
import pandas as pd
import numpy as np
from PIL import Image
import cv2
from tika import parser


def set_ck_max_to_255(cropped_dapi_img: np.ndarray, cropped_ck_img: np.ndarray, cropped_cd45_img: np.ndarray):
    """
    Adjust the brightness of provided images so that their maximum intensity values scale to 255.
    This function is used to display the images in the gallery - otherwise the signal intensity of the respective
    channels might be too low to see.

    Parameters:
        cropped_dapi_img (np.ndarray): Numpy array of the DAPI image.
        cropped_ck_img (np.ndarray): Numpy array of the CK image.
        cropped_cd45_img (np.ndarray): Numpy array of the CD45 image.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: The brightness-adjusted images for DAPI, CK, and CD45.

    """
    np.seterr(divide='ignore', invalid='ignore')

    dapi_max_255 = (cropped_dapi_img * (255 / cropped_dapi_img.max())).astype(np.uint8)
    ck_max_255 = (cropped_ck_img * (255 / cropped_ck_img.max())).astype(np.uint8)
    cd45_max_255 = (cropped_cd45_img * (255 / cropped_cd45_img.max())).astype(np.uint8)

    return dapi_max_255, ck_max_255, cd45_max_255


def save_gallery_images(path: str, df: pd.DataFrame, save_path: str, index_counter:int) -> Tuple[str, str, str, str]:
    """
    Process and save images from given paths in the dataframe with brightness adjustment. Images are from the gallery.

    Parameters:
        path (str): The root directory path containing the images.
        df (pd.DataFrame): DataFrame containing image path information.
        save_path (str): Path to save the processed images.
        index_counter (int): Index in the DataFrame to process the images for.

    Returns:
        Tuple[str, str, str, str]: Paths of the saved images.
    """

    dapi_img = np.array(Image.open(os.path.join(path, df["dapi_path"].iloc[index_counter])).convert('L'))
    ck_img = np.array(Image.open(os.path.join(path, df["ck_path"].iloc[index_counter])).convert('L'))
    cd45_img = np.array(Image.open(os.path.join(path, df["cd45_path"].iloc[index_counter])).convert('L'))

    dapi_img, ck_img, cd45_img = set_ck_max_to_255(dapi_img, ck_img, cd45_img)
    overlay_dapi_ck_img = np.dstack((dapi_img, ck_img, dapi_img))

    # New image name will be the index number since we want unique names across cases
    base_index =  str(df.index[index_counter])

    dapi_img_save_path = os.path.join(save_path, "relabel_images", f"{base_index}_1_10.png")
    cv2.imwrite(dapi_img_save_path, dapi_img)

    ck_img_save_path = os.path.join(save_path, "relabel_images", f"{base_index}_3_10.png")
    cv2.imwrite(ck_img_save_path, ck_img)

    cd45_img_save_path = os.path.join(save_path, "relabel_images", f"{base_index}_4_10.png")
    cv2.imwrite(cd45_img_save_path, cd45_img)

    overlay_img_save_path = os.path.join(save_path, "relabel_images", f"{base_index}_13_10.png")
    cv2.imwrite(overlay_img_save_path, overlay_dapi_ck_img)

    return dapi_img_save_path, ck_img_save_path, cd45_img_save_path, overlay_img_save_path


def read_text_from_gallery(file_path: str) -> pd.DataFrame:
    """
    Extracts text from a PDF file and saves it to a CSV file.
    For Windows: Checks whether apache tika is installed.

    Params:
        file_path (str): Path to the PDF file.

    Returns:
        df (pd.Dataframe): Dataframe containing the extracted text.
    """

    system_platform = platform.system().lower()

    if system_platform == "windows":
        if [["$(docker images ${apache/tika:1.28.2-full[0]} | grep ${apache/tika:1.28.2-full[1]} 2> /dev/null)" != ""]]:
            print("Docker image already exists")
        else:
            os.system("cmd /k docker run -d -p 9998:9998 apache/tika:1.28.2-full")

    # Parse the PDF file to extract text
    raw = parser.from_file(file_path)

    # Save the extracted text to a CSV file
    with open(file_path.split(".pdf")[0] + '.csv', 'w') as out:
        out.write(raw['content'].strip())

    df = pd.read_csv(file_path.split(".pdf")[0] + '.csv', delimiter="!", encoding='unicode_escape')

    return df


def get_label_and_comment(df: pd.DataFrame, df_preselection: pd.DataFrame, select: bool) \
        -> Tuple[pd.DataFrame, Optional[int], Optional[int]]:
    """
    Extracts labels and comments from a DataFrame and adds them to a second DataFrame.

    Params:
        df (pd.Dataframe): Dataframe containing the extracted text.
        df_preselection (pd.Dataframe): The DataFrame to which extracted labels and comments will be added.
        select (bool): Whether to prompt the user to enter the number of CTCs and Non-CTCs.

    Returns:
        Tuple[pd.DataFrame, Optional[int], Optional[int]]: The updated DataFrame with labels and comments,
                                                           and the optional user-specified counts of CTCs and Non-CTCs.
    """
    # Gets everything that was written out to the csv file from the gallery pdf
    df_temp = df['untitled'].to_list()

    print("Get labels and comments")
    # Get labels
    label_temp_lst = [i for i in df_temp if i.startswith('\tl')]
    label_lst = []
    for i in range(len(label_temp_lst)):
        label = label_temp_lst[i]
        label_lst.append(label.split(":")[1].split(" ")[-1])

    # Get comments
    comment_temp_lst = [i for i in df_temp if i.startswith('\tcomment')]
    comment_lst = []
    for i in range(len(comment_temp_lst)):
        comment = comment_temp_lst[i]
        comment_temp = comment.split(" ")[1:]
        comment_to_sentence = " ".join(comment_temp)
        comment_lst.append(comment_to_sentence)

    df_preselection.loc[:, 'label'] = label_lst
    df_preselection.loc[:, 'comment'] = comment_lst

    # Re-order columns, put column label and comment next to each other
    cols = df_preselection.columns.to_list()
    cols = cols[:3] + cols[-1:] + cols[3:-1]
    df_preselection = df_preselection[cols]
    print("We have the following annotations: ", Counter(df_preselection['label'].tolist()))
    if select:
        ctc_value = int(input("Enter how many CTCs you want me to include in your labeled data: "))
        print(ctc_value)
        nonctc_value = int(input("And how many Non-CTCs?: "))
        print(nonctc_value)
    else:
        ctc_value = None
        nonctc_value = None
    return df_preselection, ctc_value, nonctc_value
