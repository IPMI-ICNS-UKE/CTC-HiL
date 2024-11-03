from functools import partial
import os

import numpy as np
import pandas as pd
from typing import Tuple

from hil.utils.general_helper import create_directory


def create_folders(output_folder: str) -> None:
    concat_path = partial(os.path.join, output_folder)
    subfolders: Tuple[str, ...] = ('data', 'dataframe', 'plots', 'model',
                                   'gallery', 'gallery/relabel_images', 'gallery/relabel_pdf')
    for subfolder in map(concat_path, subfolders):
        create_directory(subfolder)


def get_most_incorrect_class(
        test_pool_df: pd.DataFrame,
        df_cluster_metrics: pd.DataFrame
) -> Tuple[int, int]:
    """
    Identify the cluster with the lowest F1 score and the most predicted incorrect class.
    """

    # Find the cluster with the lowest F1 score
    lowest_f1_score = df_cluster_metrics["cluster f1"].min()
    cluster_with_lowest_f1_score = df_cluster_metrics.loc[
        df_cluster_metrics["cluster f1"] == lowest_f1_score, "cluster name"].values[0]
    idx_lst = np.where(test_pool_df["cluster"] == cluster_with_lowest_f1_score)[0]

    # Identify incorrect predictions within the cluster
    test_preds = np.array(test_pool_df["preds"])
    cluster_test_preds = test_preds[idx_lst]
    cluster_test_labels = test_pool_df["label"][idx_lst]
    incorrect_preds_idx_lst = np.where(cluster_test_preds != cluster_test_labels)[0]
    incorrect_classes = cluster_test_preds[incorrect_preds_idx_lst]

    # Determine the most frequently incorrect class
    most_incorrect_class = np.argmax(np.bincount(incorrect_classes))
    print(f"Most predicted (incorrect) class in cluster {cluster_with_lowest_f1_score} is {most_incorrect_class}")

    return cluster_with_lowest_f1_score, most_incorrect_class
