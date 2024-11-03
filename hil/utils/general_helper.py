import os
import platform
import pathlib

from decimal import Decimal, ROUND_HALF_UP
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Union, Optional, Tuple
import sklearn as sk
import yaml
from tqdm import tqdm
import torch
from torch.cuda.amp import autocast
from torch import nn
from torch.utils.data import DataLoader
from skops.io import dump, load
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import NearestNeighbors
from sklearn.base import BaseEstimator

from hil.utils.line_plot import plot_line_plot_s1
from hil.utils.dataset import Dataset_labeled


def configure_paths():
    """
    Configure platform-specific paths.
    """
    if platform.system() == "Windows":
        print("System: Windows")
        pathlib.PosixPath = pathlib.WindowsPath
    else:
        pass


def create_directory(path: str) -> None:
    """
    Create a directory if it does not exist.

    Params:
        path (str): The path of the directory to be created.
    """
    os.makedirs(path, exist_ok=True)


def load_config(cfg_path: str):
    """
    Load configuration from a given YAML file path.

    Params:
        cfg_path (str): The path to the YAML configuration file.

    Returns:
        Dict[str, Any]: A dictionary containing the configuration data.
    """
    try:
        with open(cfg_path) as f:
            config = yaml.load(f, yaml.Loader)
        return config
    except FileNotFoundError as e:
        print(f"Config path not found: {cfg_path}")
        raise e
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file: {cfg_path}")
        raise e


def read_dataframe(case_lst: List[str],
                   path_to_df: str,
                   datatype: str,
                   ) -> pd.DataFrame:
    """
    Reads dataframes from specified directories for multiple cases and concatenates them into a single dataframe.

    This function utilizes an example path to derive the paths for other cases based on a consistent directory
    structure. The assumption is that all case-specific dataframes follow the same file path pattern, which allows
    for dynamic generation of paths using a template derived from the example path. The example path's case (such as an
    ID) is dynamically replaced with each case ID from the provided list, and either 'unlabeled' or 'labeled' is
    replaced with the specified data type. But it assumes that the unlabeled or labeled dataframe are both in the same
    directory.

    Params:
        case_lst (List[str]): A list of case identifiers, such as cartridge numbers or other IDs which are expected to
                              match the identifiers in the directory structure.
        path_to_df (str): An example absolute path to a dataframe that includes a case identifier.
                          This serves as a template for generating paths for other cases.
        datatype (str): The type of data specified, either 'labeled' or 'unlabeled', which is used to construct
                        the paths dynamically.

    Returns:
        pd.DataFrame: A concatenated pandas DataFrame containing data from all specified cases. Each dataframe is
                      read and combined into this single dataframe.
    """
    combined_df = pd.DataFrame()

    # Extract the case_id from the provided example path
    example_case_id = next((case for case in case_lst if case in path_to_df), None)

    if example_case_id:
        # Create a template path for dataframes by replacing the case_id and datatype hints
        template = path_to_df.replace(example_case_id, "{case_id}")
        template = template.replace("unlabeled", "{datatype}").replace("labeled", "{datatype}")
    else:
        raise ValueError("Example case ID not found in the example path.")

    # Iterate over case_lst to read and concatenate dataframes from each case
    for current_case in case_lst:
        current_path_to_df = template.format(case_id=current_case, datatype=datatype)

        try:
            current_df = pd.read_csv(
                current_path_to_df,
                converters={
                    'case': str,
                },
                low_memory=False
            )
            combined_df = pd.concat([combined_df, current_df], axis=0)
        except FileNotFoundError:
            print(f"Warning: Dataframe for case {current_case} not found.")

    combined_df.reset_index(drop=True, inplace=True)
    return combined_df


def extract_feature(model,
                    data_loader,
                    device):
    """
    Calculate the features of all images in the data_loader using the given model.

    Params:
        model: Trained model.
        data_loader: Data loader (torch).
        device: Your device of choice.

    Returns:
        features (List): List of all the features as numpy arrays in the same order as they were in the dataloader
                         with the length of the given dataloader.
         labels (list): List of all true labels (int) in the same order as they were in the dataloader.
    """
    features = []
    labels = []
    model.eval()
    print("Extracting features")
    for batch in tqdm(data_loader):
        images, label = batch
        images = images.to(device)
        with autocast() and torch.inference_mode():
            feature = model(images)
        features.extend(list(feature.detach().cpu().numpy()))
        if isinstance(label, torch.Tensor):
            label = label.detach().cpu().numpy()
        labels.extend(list(label))
    return features, labels


def load_features(
        root_path: str,
        img_paths: List[List[str]],
        label_list: List[int],
        model,
        device,
        batch_size: int,
        num_workers: int,
        save_path: str,
        features_file_name: str,
        labels_file_name: str
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Loads or extracts features and labels for given images using a model.

    Params:
        root_path (str): The root directory path.
        img_paths (List[str]): List of lists of string image paths. Each inner list consists of 3 paths,
                               a path for a DAPI, CK and CD45 image.
        label_list (List[int]): List of labels corresponding to image paths.
        model: Pre-trained model for feature extraction.
        device: The computing device.
        batch_size (int): Batch size for data loading.
        num_workers (int): Number of workers for data loading.
        save_path (str): Directory to save/load features and labels.
        features_file_name (str): File name for saving/loading features.
        labels_file_name (str): File name for saving/loading labels.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Features and labels as NumPy arrays.
    """

    features_path = os.path.join(save_path, f"{features_file_name}.npy")
    labels_path = os.path.join(save_path, f"{labels_file_name}.npy")

    try:
        features = np.load(features_path)
        labels = np.load(labels_path)
        print(f"Loaded features and labels: {features_file_name}, {labels_file_name}")
    except FileNotFoundError:
        print(f"File not found, extracting feature data: {features_file_name}, {labels_file_name}")

        data_set = Dataset_labeled(
            root_path,
            img_paths=img_paths,
            label_list=label_list,
            normalize_ctc=True
        )

        data_loader = DataLoader(
            dataset=data_set,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False,
            drop_last=False
        )

        features, labels = extract_feature(model, data_loader, device)

        features = np.array(features)
        labels = np.array(labels)

        np.save(features_path, features)
        np.save(labels_path, labels)

    return features, labels


def load_model(model: nn.Module, device: int, path: str) -> nn.Module:
    """
    Load model weights.
    Returns: The model with loaded weights, moved to the specified device.
    """
    map_location = f'cuda:{device}'
    weights = torch.load(path, map_location=map_location, weights_only=True)
    weights = {key.replace('backbone.', ''): weight for key, weight in weights.items() if 'backbone.' in key}
    model.load_state_dict(weights, strict=True)
    model.to(device)
    print("Model to Device: ", device)
    return model


def map_unique_elements_with_counts(lst: List[Any]) -> Dict[Any, int]:
    """
    Maps unique elements of the list to their respective counts.
    Params:
        lst (List[Any]): A list of elements.
    Returns:
        Dictionary: A dictionary where keys are unique elements from the list; values are their respective counts.
    """
    mapping = dict(zip(np.unique(lst), np.unique(lst, return_counts=True)[-1]))
    return mapping


def define_and_fit_classifier(
        opt: List[Dict[str, Dict[str, Any]]],
        features: np.ndarray,
        labels: np.ndarray,
        save_path: str,
        file_name: str
) -> BaseEstimator:
    """
    Load a classifier from a specified save path, or fit and save a new classifier if none exists.

    Params:
        opt List[Dict[str, Dict[str, Any]]]: Dictionary containing classifier type and parameters.
                                             {'classifier': <classifier_name>, 'params': <parameters_dict>}
        features (np.ndarray): The input features used to train the classifier.
        labels (np.ndarray): The target labels corresponding to the input features.
        save_path (str): The directory path where the classifier is saved or loaded from.
        file_name (str): The name of the file to save or load the classifier from.

    Returns:
        BaseEstimator: A trained classifier object, either loaded or newly created.
    """
    try:
        predictor = load(os.path.join(save_path, f"{file_name}.skops"))
    except FileNotFoundError:
        classifier_name = opt[0]['classifier']
        params = opt[0]['params']
        classifier = getattr(__import__('sklearn.svm', fromlist=[classifier_name]), classifier_name)
        predictor = classifier(**params)
        predictor.fit(features, labels)
        dump(predictor, os.path.join(save_path, f"{file_name}.skops"))
    return predictor


def calculate_recall_precision(labels: np.ndarray, preds: np.ndarray) -> (float, float):
    """
    Calculate and return recall and precision based on provided labels and predictions.
    Formula from: https://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html

    Params:
        labels (np.ndarray): True labels.
        preds (np.ndarray): Predicted labels.

    Returns:
        recall (float): Rounded recall value
        precision (float): Rounded precision value
        """
    matrix = confusion_matrix(labels, preds).ravel()
    if len(matrix) == 1:  # All Labels and Predictions are the same
        if labels.tolist() == preds.tolist():
            if labels.tolist()[0] == 1:
                tp = matrix[0]  # All are True Positives
                tn = fp = fn = 0
            if labels.tolist()[0] == 0:
                tn = matrix[0]  # All are True Negatives
                tp = fn = fp = 0
    else:
        tn, fp, fn, tp = matrix.ravel()

    recall = tp / (fn + tp) if (fn + tp) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0

    # ratio between how much were correctly classified as negative to how much was actually negative
    # specificity = np.round(tn / (fp + tn), 4)
    # ratio between how much were correctly identified as positive to how much were actually positive
    # sensitivity = np.round(tp / (fn + tp), 4)  # same as recall

    return np.round(recall, 4), np.round(precision, 4)


def calculate_specificity(labels: np.ndarray, preds: np.ndarray) -> float:
    tn = sum((y_true == 0 and y_pred == 0) for y_true, y_pred in zip(labels, preds))
    fp = sum((y_true == 0 and y_pred == 1) for y_true, y_pred in zip(labels, preds))
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    return np.round(specificity, 4)


def add_preds_to_df(df: pd.DataFrame, preds: np.ndarray, labels: np.ndarray, data_type_string: str) -> pd.DataFrame:
    """
    Add prediction results to a DataFrame with the appropriate label and prediction strings.

    Params:
        df (pd.DataFrame): The DataFrame to which predictions and labels will be added.
        preds (np.ndarray): Predicted labels.
        labels (np.ndarray): True labels.
        data_type_string (str): String indicating the type of data ('unlabeled train' or other).

    Returns:
        pd.DataFrame: DataFrame with added string predictions and labels as new columns.
    """
    df["preds"] = preds
    df["label_string"] = labels
    if data_type_string == "unlabeled train":
        df_added = df.replace({'label_string': {10: "Unlabeled"}})
        df_added["preds_string"] = df_added["preds"].copy()
        df_added = df_added.replace({'preds_string': {0: "Non-CTC (pred)", 1: "CTC (pred)"}})
    else:
        df_added = df.replace({'label_string': {0: "Non-CTC GT (" + data_type_string + ")",
                                                1: "CTC GT (" + data_type_string + ")"}})
        # Get indices where false predictions happened
        false_preds_idx_lst = np.where(df_added["label"] != df_added["preds"])[0]
        df_added["preds_string"] = df_added["preds"].copy()
        df_added = df_added.replace({'preds_string': {0: "Non-CTC (correct)", 1: "CTC (correct)"}})
        df_added.iloc[false_preds_idx_lst] = df_added.iloc[false_preds_idx_lst].replace(
            {'preds_string': {"Non-CTC (correct)": "Non-CTC (but GT:CTC)", "CTC (correct)": "CTC (but GT:Non-CTC)"}})
    return df_added


def knn_cluster_neighbors(fit_umap_features: np.ndarray, find_neighbors_of_umap_features: np.ndarray) -> np.ndarray:
    """
    Find the n nearest neighbors for a set of features using the k-Nearest Neighbors algorithm.

    Params:
        fit_umap_features (np.ndarray): Features to fit the NearestNeighbors model.
        find_neighbors_umap_features (np.ndarray): Features for which the nearest neighbors will be found.

    Returns:
        np.ndarray: Indices of the nearest neighbors.
    """
    knn = NearestNeighbors(n_neighbors=1, metric='euclidean')
    knn.fit(fit_umap_features)
    dist_knn, indices_knn = knn.kneighbors(find_neighbors_of_umap_features)
    return indices_knn


def assign_clusters(
    labeled_test_and_unlabeled_train_umap_features: np.ndarray,
    df_labeled_test_and_unlabeled_train: pd.DataFrame,
    labeled_train_umap_features: np.ndarray,
    assign_background_cluster_no_nearest_cluster: bool
) -> np.ndarray:
    """
    Assigns clusters to labeled training data and optionally background clusters.

    Returns:
        np.ndarray: Array of cluster assignments.
    """
    # Find nearest neighbors
    indices_knn = knn_cluster_neighbors(
        fit_umap_features=labeled_test_and_unlabeled_train_umap_features,
        find_neighbors_of_umap_features=labeled_train_umap_features
    )

    # Create a list of assigned clusters
    cluster_lst = [df_labeled_test_and_unlabeled_train["cluster"].iloc[idx[0]] for idx in indices_knn]

    # Optionally assign background clusters
    if assign_background_cluster_no_nearest_cluster:
        cluster_lst = assign_background_clusters(
            cluster_lst,
            df_labeled_test_and_unlabeled_train,
            labeled_test_and_unlabeled_train_umap_features,
            labeled_train_umap_features
        )
    return np.array(cluster_lst)

def assign_background_clusters(
    cluster_lst: list,
    df_labeled_test_and_unlabeled_train: pd.DataFrame,
    labeled_test_and_unlabeled_train_umap_features: np.ndarray,
    labeled_train_umap_features: np.ndarray
) -> list:
    """
    Assigns background clusters for data points labeled as -1.

    Returns:
        list: Updated list of cluster assignments.
    """

    # Convert data to numpy array if not already
    cluster_arr = np.array(cluster_lst)
    # Indices of the labeled train data that need background assignment
    background_indices = np.where(cluster_arr == -1)[0]

    # Indices of test and unlabeled train data without background
    without_background_indices = np.where(df_labeled_test_and_unlabeled_train["cluster"] != -1)[0]

    # Calculate nearest neighbors for background points
    indices_knn = knn_cluster_neighbors(
        fit_umap_features=labeled_test_and_unlabeled_train_umap_features[without_background_indices],
        find_neighbors_of_umap_features=labeled_train_umap_features[background_indices]
    )

    for idx, current_idx in enumerate(indices_knn):
        # Assign the nearest valid cluster to the background point
        nearest_neighbor_idx = without_background_indices[current_idx[0]]
        cluster_lst[background_indices[idx]] = df_labeled_test_and_unlabeled_train.iloc[nearest_neighbor_idx]["cluster"]

    return cluster_lst


def general_metrics(
        experiment_name: str,
        opt: List[Dict[str, Dict[str, Any]]],
        train_set: np.ndarray,
        test_set: np.ndarray,
        unlabeled_train_set: Optional[np.ndarray],
        train_labels: np.ndarray,
        test_labels: np.ndarray,
        train_unlabels: Optional[np.ndarray],
        process_train_unlabeled_data: bool,
        df_unlabeled_train: Optional[pd.DataFrame],
        model_save_path: str,
        run: Optional[int] = None,
        loop: Optional[Union[str, int]] = None
) -> Union[Tuple[pd.DataFrame, np.ndarray, pd.DataFrame], Tuple[pd.DataFrame, np.ndarray]]:
    """
    Caluclate metrcis over train and test sets, and optionally process unlabeled train data.

    Params:
        experiment_name (str): Name of experiment, e.g. either cluster_specific or random
        opt List[Dict[str, Dict[str, Any]]]: Dictionary containing classifier type and parameters.
                                             {'classifier': <classifier_name>, 'params': <parameters_dict>}
        train_set (np.ndarray): Features of train set (e.g. PCA features)
        test_set (np.ndarray): Features of test set (e.g. PCA features)
        unlabeled_train_set (Optional[np.ndarray]): Features of unlabeled train set (e.g., PCA features)
        train_labels (np.ndarray): Labels for train set
        test_labels (np.ndarray): Labels for test set
        train_unlabels (Optional[np.ndarray]): Labels for unlabeled train set (e.g. "10" for "it could either be a
                                               CTC (label: 1) or a non-CTC (label:0).
        process_train_unlabeled_data (bool): Flag to process train unlabeled data.
        df_unlabeled_train (Optional[pd.DataFrame]): DataFrame of unlabeled train set.
        model_save_path (str): Path to save model.
        run (Optional[int]): Current run. Since we can perform multiple runs (each run has n number of loops).
        loop (Optional[Union[str, int]]): Current loop (The current loop can be the "initialization" or loop 1, 2, .. etc.)

    Returns:
        Union[Tuple[pd.DataFrame, np.ndarray, pd.DataFrame], Tuple[pd.DataFrame, np.ndarray]]: DataFrame containing
        general metrics, test predictions, and optionally DataFrame of unlabeled train with added columns for
        predictions in string format.
    """
    # Fit the classifier or load classifier from folder based on provided path
    predictor = define_and_fit_classifier(
        opt=opt,
        features=train_set,
        labels=train_labels,
        save_path=model_save_path,
        file_name="classifier_model"
    )

    train_preds = predictor.predict(train_set)
    test_preds = predictor.predict(test_set)

    if process_train_unlabeled_data and unlabeled_train_set is not None and train_unlabels is not None:
        if loop == "init":
            train_unlabeled_preds = predictor.predict(unlabeled_train_set)
            df_unlabeled_train = add_preds_to_df(
                df=df_unlabeled_train,
                preds=train_unlabeled_preds,
                labels=train_unlabels,
                data_type_string="unlabeled train"
            )

    acc_train = sk.metrics.balanced_accuracy_score(train_labels, train_preds)
    f1_score_train = sk.metrics.f1_score(train_labels, train_preds)
    recall_train, precision_train = calculate_recall_precision(labels=train_labels, preds=train_preds)

    acc_test = sk.metrics.balanced_accuracy_score(test_labels, test_preds)
    f1_score_test = sk.metrics.f1_score(test_labels, test_preds)
    recall_test, precision_test = calculate_recall_precision(labels=test_labels, preds=test_preds)

    general_metrics_dict = {
        'experiment name': experiment_name,
        'run': run if run is not None else '',
        'loop': loop if loop is not None else '',
        'train balanced acc': acc_train,
        'train f1 score': f1_score_train,
        'train recall': recall_train,
        'train precision': precision_train,
        'test balanced acc': acc_test,
        'test f1 score': f1_score_test,
        'test recall': recall_test,
        'test precision': precision_test
    }
    df_general_metrics = pd.DataFrame([general_metrics_dict])

    if process_train_unlabeled_data:
        return df_general_metrics, test_preds, df_unlabeled_train
    else:
        return df_general_metrics, test_preds


def cluster_metrics(
        experiment_name: str,
        cluster_lst: np.ndarray,
        cluster_preds: Union[pd.Series, np.ndarray],
        labels: np.ndarray,
        preds: np.ndarray,
        known_labels=None,
        run: Optional[int] = None,
        loop: Optional[Union[str, int]] = None
) -> pd.DataFrame:
    """
    Calculate metrics for each cluster in the given dataframe.

    Params:
        experiment_name (str): Name of experiment, e.g. either cluster_specific or random
        cluster_lst (np.ndarray): Array of unique clusters (based on df).
        cluster_preds (Union[pd.Series, np.ndarray]): Cluster predictions.
        labels (np.ndarray): True labels. Must have the same length as df.
        preds: Predicted labels. Must have the same length as df.
        known_labels: Known labels are here the labels of your classes. Here usually: 0 (= non-CTC) and 1 (= CTC).
        run (Optional[int]): Current run. Since we can perform multiple hil runs (each run has n number of loops).
        loop (Optional[Union[str, int]]): Current loop (The current loop can be the "initialization" or loop 1, 2, .. etc.)

    Returns:
        pd.DataFrame: DataFrame containing the metrics for each cluster.
    """
    if known_labels is None:
        known_labels = [0, 1]
    df_metrics = pd.DataFrame()
    for i, cluster in enumerate(cluster_lst):
        # cluster_idx_lst = df.loc[df['cluster'] == cluster].index.to_list()
        cluster_idx_lst = cluster_preds[cluster_preds == cluster].index.to_list()

        recall, precision = calculate_recall_precision(labels=labels[cluster_idx_lst], preds=preds[cluster_idx_lst])
        acc = sk.metrics.balanced_accuracy_score(labels[cluster_idx_lst], preds[cluster_idx_lst])
        f1_score = sk.metrics.f1_score(labels[cluster_idx_lst], preds[cluster_idx_lst],
                                       labels=known_labels, zero_division=0)
        specificity = calculate_specificity(labels[cluster_idx_lst], preds[cluster_idx_lst])
        # if F1 score is 0 but all predictions are correct:
        if f1_score == 0 and set(labels[cluster_idx_lst]) == {0} and set(preds[cluster_idx_lst]) == {0}:
            f1_score = specificity

        metrics_dict = {
            "experiment name": experiment_name,
            'run': run if run is not None else '',
            'loop': loop if loop is not None else '',
            'cluster name': cluster,
            'cluster balanced acc': acc,
            'cluster f1': f1_score,
            'cluster recall': recall,
            'cluster precision': precision
        }
        df_metrics_temp = pd.DataFrame([metrics_dict])
        df_metrics = pd.concat([df_metrics, df_metrics_temp], ignore_index=True)

    return df_metrics


def update_df_with_mean_and_std(n_runs: int, df: pd.DataFrame, decimal_place: int) -> pd.DataFrame:
    """
    Update DataFrame by computing cluster F1 scores and std over multiple hil runs.

    Parameters:
        n_runs (int): Total number of hil runs.
        df (pd.DataFrame): The DataFrame containing cluster results.
        decimal_place (int): Number of decimal places -> for mean and standard deviation.

    Returns:
        pd.DataFrame: A DataFrame containing the experiment name, cluster name, loop,
                      F1 scores across runs, and their mean and standard deviation.
    """

    df_cluster_results_for_plotting = pd.DataFrame()

    # Copy the three columns into new dataframe
    df_temp_run = df["run"] == 1
    df_cluster_results_for_plotting["experiment name"] = df.loc[df_temp_run, "experiment name"]
    df_cluster_results_for_plotting["cluster name"] = df.loc[df_temp_run, "cluster name"]
    df_cluster_results_for_plotting["loop"] = df.loc[df_temp_run, "loop"]

    # Iterate over hil-runs and add cluster F1 scores to the new dataframe
    for run in range(n_runs):
        current_run = run + 1
        column_name = f"cluster f1 run {current_run}"
        df_temp_current_run = df["run"] == current_run
        df_cluster_results_for_plotting[column_name] = df.loc[df_temp_current_run, "cluster f1"].to_list()

    # Calculate mean and std of F1 scores
    f1_columns = [f"cluster f1 run {i + 1}" for i in range(n_runs)]
    mean = np.round(df_cluster_results_for_plotting[f1_columns].mean(axis=1), decimal_place)
    std = np.round(df_cluster_results_for_plotting[f1_columns].std(axis=1), decimal_place)
    df_cluster_results_for_plotting["f1 mean"] = mean
    df_cluster_results_for_plotting["f1 std"] = std

    return df_cluster_results_for_plotting


def save_results(
        n_runs: int,
        save_path: str,
        experiment_name: str,
        df_general_results: pd.DataFrame,
        df_cluster_results: pd.DataFrame,
        df_mc_general_results: Optional[pd.DataFrame] = None,
        df_mc_cluster_results: Optional[pd.DataFrame] = None
) -> Union[Tuple[pd.DataFrame, pd.DataFrame], pd.DataFrame]:
    """
    Save general and cluster results to Excel files. If provided, it also saves Monte Carlo cross-validation results.

    Params:
        n_runs (int): Total number of hil runs.
        save_path (str): The directory path where results will be saved.
        experiment_name (str): Name of experiment, e.g. either cluster_specific or random
        df_general_results (pd.DataFrame): DataFrame containing general metrics.
        df_cluster_results (pd.DataFrame): DataFrame containing cluster metrics.
        df_mc_general_results (Optional[pd.DataFrame]): DataFrame for MC general metrics.
        df_mc_cluster_results (Optional[pd.DataFrame]): DataFrame for MC cluster metrics.

    Returns:
        pd.DataFrame: DataFrame prepared for plotting from cluster results. If MC provided, then it additionally returns
                      cluster results for mc experiments.
    """

    df_general_results.to_excel(
        os.path.join(save_path, f"{experiment_name}_general_metrics_results.xlsx")
    )
    df_cluster_results.to_excel(
        os.path.join(save_path, f"{experiment_name}_cluster_metrics_results.xlsx")
    )

    df_cluster_results_for_table = update_df_with_mean_and_std(
        n_runs=n_runs, df=df_cluster_results, decimal_place=3
    )
    df_cluster_results_for_table.to_excel(
        os.path.join(save_path, f"{experiment_name}_cluster_results_for_table.xlsx")
    )

    df_cluster_results_for_plotting = round_mean_and_std(df_cluster_results_for_table)

    df_cluster_results_for_plotting.to_excel(
        os.path.join(save_path, f"{experiment_name}_cluster_results_for_plotting.xlsx")
    )

    if df_mc_general_results is not None and df_mc_cluster_results is not None:
        df_mc_general_results.to_excel(
            os.path.join(save_path, f"{experiment_name}_mc_general_metrics_results.xlsx")
        )
        df_mc_cluster_results.to_excel(
            os.path.join(save_path, f"{experiment_name}_mc_cluster_metrics_results.xlsx")
        )

        df_mc_cluster_results_for_table = update_df_with_mean_and_std(
            n_runs=n_runs, df=df_mc_cluster_results, decimal_place=3
        )

        df_mc_cluster_results_for_table.to_excel(
            os.path.join(save_path, f"{experiment_name}_mc_cluster_results_for_table.xlsx")
        )

        df_mc_cluster_results_for_plotting = round_mean_and_std(df_mc_cluster_results_for_table)

        df_mc_cluster_results_for_plotting.to_excel(
            os.path.join(save_path, f"{experiment_name}_mc_cluster_results_for_plotting.xlsx")
        )

        return df_mc_cluster_results_for_plotting, df_cluster_results_for_plotting

    return df_cluster_results_for_plotting


def round_mean_and_std(df: pd.DataFrame) -> pd.DataFrame:
    mean_lst = df["f1 mean"].to_list()
    rounded_mean_lst = [
        float(Decimal(str(value)).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP))
        for value in mean_lst
    ]

    std_lst = df["f1 std"].to_list()
    rounded_std_lst = [
        float(Decimal(str(value)).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP))
        for value in std_lst
    ]

    # update columns
    df["f1 mean"] = rounded_mean_lst
    df["f1 std"] = rounded_std_lst

    return df


def plot_results(
        experiment_name: str,
        df_cluster_results_for_plotting: pd.DataFrame,
        save_path: str,
        df_mc_cluster_results_for_plotting=None
):
    """
    Plot results for different experimental scenarios (cluster-specific, random, mc).
    """
    title = "experiment"
    file_name = "experiment_f1_score"
    if experiment_name == "cluster_specific":
        title = "cluster-specific experiment"
        file_name = experiment_name + "_f1_score"
    elif experiment_name == "random":
        title = experiment_name + " experiment"
        file_name = experiment_name + "_f1_score"

    if df_mc_cluster_results_for_plotting is not None:
        plot_line_plot_s1(
            df_results=df_mc_cluster_results_for_plotting,
            title=f"MC {title}",
            file_name=f"mc_{file_name}",
            save_path=os.path.join(save_path, "plots")
        )

    plot_line_plot_s1(
        df_results=df_cluster_results_for_plotting,
        title=title,
        file_name=file_name,
        save_path=os.path.join(save_path, "plots")
    )
