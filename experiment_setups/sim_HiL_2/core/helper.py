import os
import random
from functools import partial

import numpy as np
import pandas as pd
from typing import Tuple, List


def create_simulation_experiment_folders(output_folder: str) -> None:
    concat_path = partial(os.path.join, output_folder)
    subfolders: Tuple[str, ...] = ('plots', 'model', 'data')
    for subfolder in map(concat_path, subfolders):
        os.makedirs(subfolder, exist_ok=True)


def split_randomly_by_percentage(
        lst: pd.Series,
        cluster: int,
        seed: int,
        percentage: float
) -> Tuple[List[int], List[int]]:
    # lst = cluster list
    idx_lst = np.where(lst == cluster)[0].tolist()
    random.Random(seed).shuffle(idx_lst)
    random.seed(seed)
    percentage_idx_lst = random.sample(idx_lst, k=round(len(idx_lst) * percentage))
    remaining_percentage_idx_lst = list(set(idx_lst) - set(percentage_idx_lst))
    return percentage_idx_lst, remaining_percentage_idx_lst



