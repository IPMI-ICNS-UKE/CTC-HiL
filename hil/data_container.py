import os
from typing import Dict, List
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class DataContainer:
    """
    A container for managing data loading from file paths.

    Attributes:
        path_to_input_folder (str): The path to the input folder containing data files.
    """
    path_to_input_folder: str

    def __post_init__(self) -> None:
        """
        Initializes a dictionary to store loaded data.
        """
        self.data: Dict[str, np.ndarray] = {}

    def load_data(self) -> None:
        """
        Loads data from CSV and NPY files based on predefined keys.

        Data is loaded into self.data dictionary with the key naming convention:
        - 'df_' for pandas DataFrames
        - Otherwise, numpy arrays
        """
        data_keys: List[List[str]] = [
            # Dataframes
            ['df_labeled_train', 'df_labeled_test_and_unlabeled_train', 'df_labeled_test', 'df_unlabeled_train'],
            # PCA features
            ['labeled_train_pca_features', 'labeled_test_pca_features', 'unlabeled_train_pca_features'],
            # UMAP features
            ['labeled_train_umap_features', 'labeled_test_and_unlabeled_train_umap_features',
             'labeled_test_umap_features', 'unlabeled_train_umap_features']
        ]
        for group_keys in data_keys:
            for key in group_keys:
                file_path = os.path.join(self.path_to_input_folder, key)
                if key.startswith('df_'):
                    self.data[key] = pd.read_csv(file_path)
                else:
                    self.data[key] = np.load(file_path + '.npy')

    def add_column_to_df_labeled_train(self, new_column_data, column_name: str) -> None:
        """
        Adds a new column to the 'df_labeled_train' dataframe.

        Params:
            new_column_data: The data for the new column.
            column_name: The name of the new column to add.
        """
        if 'df_labeled_train' in self.data:
            self.data['df_labeled_train'][column_name] = new_column_data
        else:
            raise ValueError("df_labeled_train is not loaded in data container.")

    def __getattr__(self, name: str):
        """
        Params:
            name(str): The name of the attribute requested.
        Returns:
            The corresponding data from the internal storage.
        """
        if name in self.data:
            return self.data[name]
        else:
            raise AttributeError(f"'DataContainer' object has no attribute '{name}'")

