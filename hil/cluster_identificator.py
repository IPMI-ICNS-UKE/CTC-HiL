import numpy as np
import pandas as pd
from tabulate import tabulate
from hdbscan import HDBSCAN
from typing import Tuple, List, Union

from hil.utils.general_helper import map_unique_elements_with_counts


def align_clusters(
        cluster_ids: List[int],
        cluster_counts: List[int],
        display_clusters: np.ndarray
) -> List[int]:
    """
    Align cluster counts with specified clusters.

    Parameters:
        cluster_ids (List[int]): List of unique cluster identifiers/ names.
        cluster_counts (List[int]): List of counts corresponding to each cluster ID.
        display_clusters (List[int]): List of clusters for which counts need to be aligned.

    Returns:
        List[int]: Cluster counts aligned with clusters.
    """
    cluster_dict = dict(zip(cluster_ids, cluster_counts))
    return [cluster_dict.get(cluster, 0) for cluster in display_clusters]


class ClusterIdentificator:

    @staticmethod
    def find_clusters(umap_features: np.ndarray, clusterer: HDBSCAN):
        """
        Apply clustering algorithm to UMAP features.

        Params:
            umap_features (np.ndarray): The defined hdbscan clusterer will be applied on these umap features.
            clusterer (HDBSCAN): The HDBSCAN clusterer. HDBSCAN params are defined beforehand.

        Returns:
            np.ndarray: Predicted clusters.
        """
        cluster_pred = clusterer.fit_predict(umap_features)
        return cluster_pred

    @staticmethod
    def assign_clusters(
            cluster_pred: np.ndarray,
            df_labeled_test: pd.DataFrame,
            df_unlabeled_train: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
        """
        Assign cluster predictions to labeled test and unlabeled train dataframes.

        Params:
            cluster_pred (np.ndarray): Cluster predictions.
            df_labeled_test (pd.DataFrame): Dataframe of labeled test data.
            df_unlabeled_train (pd.DataFrame): Dataframe of unlabeled training data.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, np.ndarray]: Updated dataframes with cluster assignment.
        """
        df_labeled_test["cluster"] = cluster_pred[:len(df_labeled_test)]
        df_unlabeled_train["cluster"] = cluster_pred[len(df_labeled_test):]

        return df_labeled_test, df_unlabeled_train, cluster_pred

    @staticmethod
    def print_cluster_summary(
            df_labeled_test: pd.DataFrame,
            df_unlabeled_train: pd.DataFrame
    ) -> None:
        """
        Print a summary table of cluster distribution across datasets.

        Params:
            df_labeled_test (pd.DataFrame): Dataframe of labeled test data with clusters.
            df_unlabeled_train (pd.DataFrame): Dataframe of unlabeled train data with clusters.
        """
        cluster_pred = np.concatenate((df_labeled_test["cluster"].to_list(), df_unlabeled_train["cluster"].to_list()))
        display_clusters = np.unique(cluster_pred)

        cluster_pred_ids, cluster_pred_counts = map(lambda x: x.tolist(), np.unique(cluster_pred, return_counts=True))

        test_cluster_ids, test_cluster_counts = map(lambda x: x.tolist(),
                                                    np.unique(df_labeled_test.cluster, return_counts=True))

        train_unlab_ids, train_unlab_cluster_counts = map(lambda x: x.tolist(),
                                                          np.unique(df_unlabeled_train.cluster, return_counts=True))

        cluster_pred_counts = align_clusters(cluster_pred_ids, cluster_pred_counts, display_clusters)
        test_cluster_counts = align_clusters(test_cluster_ids, test_cluster_counts, display_clusters)
        train_unlab_cluster_counts = align_clusters(train_unlab_ids, train_unlab_cluster_counts, display_clusters)

        table = zip(display_clusters, cluster_pred_counts, test_cluster_counts, train_unlab_cluster_counts)

        print(tabulate(table,
                       headers=['Cluster', 'total datapoints', 'test datapoints', 'train unlab datapoints'],
                       tablefmt='orgtbl'))

    @staticmethod
    def identify_test_false_predictions(
            labels: Union[pd.Series, np.ndarray],
            preds: Union[pd.Series, np.ndarray],
            cluster_preds: Union[pd.Series, np.ndarray]
    ) -> None:

        """
        Identify false predictions in your data and print their clusters.

        Params: All params must have the same length.
            labels (List[int]): True labels.
            preds (List[int]): Predicted labels.
            cluster_preds (np.ndarray): Cluster predictions.
       """
        test_false_predictions_idx_lst = [i for i, (label, pred) in enumerate(zip(labels, preds)) if label != pred]
        print("test misclassifications in clusters:", map_unique_elements_with_counts(
            cluster_preds[test_false_predictions_idx_lst]))
