import os
from functools import partial

import pandas as pd
import numpy as np
from typing import Dict, Any, Union, Tuple
import random
import sklearn as sk
from sklearn.model_selection import train_test_split

from hil.utils.general_helper import define_and_fit_classifier


def create_simulation_experiment_folders(output_folder:str) -> None:
    """
    Creates subdirectories for a simulation experiment within the specified output folder.
    Here: This function will create 'plots', 'model', and 'cv' subdirectories if they do not already exist.

    Params:
        output_folder (str): The path to the main output folder where the subdirectories will be created.
    """
    concat_path = partial(os.path.join, output_folder)
    subfolders: Tuple[str, ...] = ('plots', 'model', 'cv', 'data')
    for subfolder in map(concat_path, subfolders):
        os.makedirs(subfolder, exist_ok=True)


def monte_carlo_cross_validation_of_training_pool(
        classifier_opt: Dict[str, Any],
        train_labels: np.ndarray,
        labeled_train_pca_features: np.ndarray,
        df: pd.DataFrame,
        model_save_path: str
) -> pd.DataFrame:
    """
    Performs Monte Carlo cross-validation on a training pool using a specified classifier.

    This function repeatedly splits the data into training and Monte Carlo test subsets, trains a classifier,
    and predicts on the test subsets. The results are collected into a DataFrame.

    Params:
        classifier_opt (Dict[str, Any]): Dictionary containing the classifier type and its (hyper)parameters.
        train_labels (np.ndarray): Array of training labels.
        labeled_train_pca_features (np.ndarray): PCA feature representation of the training data.
        df (pd.DataFrame): DataFrame of the training pool.
        model_save_path (str): Path to save the trained classifier models.

    Returns:
        pd.DataFrame: DataFrame containing Monte Carlo predictions for each test subset.
        """
    n_repeats = 100
    random.seed(42)
    random_seed_lst = random.sample(range(1, 200), n_repeats)
    df_mc = pd.DataFrame()
    for current_repeat in range(n_repeats):
        num = current_repeat + 1
        random_seed = random_seed_lst[current_repeat]
        train_indices, mc_indices = train_test_split(
            df.index,
            test_size=0.1,
            random_state=random_seed
        )
        current_labeled_train_pca_features = labeled_train_pca_features[train_indices]
        current_train_labels = train_labels[train_indices]
        current_mc_pca_features = labeled_train_pca_features[mc_indices]
        # current_mc_labels = train_labels[mc_indices]
        current_df_mc = df.iloc[mc_indices].copy()
        predictor = define_and_fit_classifier(
            opt=classifier_opt,
            features=current_labeled_train_pca_features,
            labels=current_train_labels,
            save_path=model_save_path,
            file_name=f"classifier_model_{num}"
        )

        mc_preds = predictor.predict(current_mc_pca_features)
        current_df_mc["preds"] = mc_preds
        df_mc = pd.concat([df_mc, current_df_mc], ignore_index=True)

    return df_mc


def general_metrics_of_monte_carlo_results(
        experiment_name: str,
        run: int,
        loop: Union[str, int],
        df_mc: pd.DataFrame
) -> pd.DataFrame:
    """
    Computes general metrics for Monte Carlo simulation results and returns them as a DataFrame.

    This function evaluates the F1 score of the predictions from a Monte Carlo cross-validation and constructs
    a summary DataFrame with the experiment details and metrics (here: F1 score).

    Params:
        experiment_name (str): The name of the experiment.
        run (int): The current run.
        loop (Union[str, int]): The current loop which can be a string or an integer.
        df_mc (pd.DataFrame): DataFrame containing true labels and predicted labels ('label' and 'preds' columns).

    Returns:
        pd.DataFrame: A DataFrame containing the general metrics of the Monte Carlo results.
        """

    f1_score_mc = sk.metrics.f1_score(df_mc["label"], df_mc["preds"])
    general_metrics_dict = {'experiment name': experiment_name,
                            'run': run,
                            'loop': loop,
                            'mc f1 score': f1_score_mc}
    df_mc_general_metrics = pd.DataFrame([general_metrics_dict])
    return df_mc_general_metrics
