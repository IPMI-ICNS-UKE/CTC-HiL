import os

import pandas as pd
import numpy as np
from tabulate import tabulate
from pprint import pformat
import hdbscan

from hil.utils.general_helper import load_config, configure_paths, create_directory, general_metrics, cluster_metrics
from hil.utils.cluster_plot import plot_cluster_contours_in_hdbscan
from hil.cluster_identificator import ClusterIdentificator
from hil.data_container import DataContainer
from hil.utils.logger import Logger


def main():
    configure_paths()

    # Load experiment configuration from config file
    cfg_path = "configs/cluster_analysis_cfg.yml"
    config = load_config(cfg_path)

    # HDBSCAN
    clusterer = hdbscan.HDBSCAN(**config["HDBSCAN_parameters"])

    # Prepare directories for saving results
    path_to_data = config["general_paths"]["path_to_data_folder"]
    save_path = config["general_paths"]["path_to_results_folder"]
    save_path = os.path.join(save_path, "clustering_analysis_results")
    model_path = os.path.join(save_path, "model")
    plot_path = os.path.join(save_path, "plots")
    create_directory(save_path)
    create_directory(plot_path)
    create_directory(model_path)

    # Setup logger
    log_file_path = os.path.join(save_path, "clustering.log")
    logger = Logger(log_file_path=log_file_path)
    logger.info(f"Logger configured to write to {log_file_path}.")
    logger.debug(f"Initializing Cluster analysis with config: {pformat(config)}")

    # For reproducibility of figure 2B (left: clustering image), load data from data container
    data_container = DataContainer(
        path_to_input_folder=path_to_data
    )

    data_container.load_data()

    # Get cluster predictions
    cluster_pred = ClusterIdentificator.find_clusters(
        umap_features=data_container.data["labeled_test_and_unlabeled_train_umap_features"],
        clusterer=clusterer
    )

    # Cluster predictions are added to corresponding dataframes
    df_labeled_test, df_unlabeled_train, cluster_pred = ClusterIdentificator.assign_clusters(
        cluster_pred=cluster_pred,
        df_labeled_test=data_container.data["df_labeled_test"],
        df_unlabeled_train=data_container.data["df_unlabeled_train"],
    )

    # Print a tabular overview of how many data points are in the respective clusters
    ClusterIdentificator.print_cluster_summary(
        df_labeled_test=df_labeled_test,
        df_unlabeled_train=df_unlabeled_train
    )

    extra_args = {
        "unlabeled_train_set": data_container.data["unlabeled_train_pca_features"],
        "train_unlabels": df_unlabeled_train["label"],
        "df_unlabeled_train": df_unlabeled_train,
    }

    # Compute general metrics
    result = general_metrics(
        experiment_name="clustering",
        opt=config["OPT"],
        train_set=data_container.data["labeled_train_pca_features"],
        test_set=data_container.data["labeled_test_pca_features"],
        train_labels=data_container.data["df_labeled_train"]["label"],
        test_labels=df_labeled_test["label"],
        process_train_unlabeled_data=True,
        model_save_path=model_path,
        **extra_args
    )

    df_general_metrics, test_preds, df_unlabeled_train = result

    # Compute cluster metrics
    df_cluster_metrics = cluster_metrics(
        experiment_name="clustering",
        cluster_lst=np.unique(df_labeled_test["cluster"]),
        cluster_preds=df_labeled_test["cluster"],
        labels=df_labeled_test["label"],
        preds=test_preds
    )

    # Print overview of cluster metrics in tabular format
    print(tabulate(df_cluster_metrics, headers='keys', tablefmt='psql'))

    # Prints how many test misclassifications in each cluster
    ClusterIdentificator.identify_test_false_predictions(
        labels=df_labeled_test["label"],
        preds=test_preds,
        cluster_preds=df_labeled_test["cluster"]
    )

    df_labeled_test_and_unlabeled_train = pd.concat([df_labeled_test, df_unlabeled_train], ignore_index=True)

    # Plot clusters with contours
    plot_cluster_contours_in_hdbscan(
        train_unlabels=df_unlabeled_train["label"],
        test_labels=df_labeled_test["label"],
        test_preds=test_preds,
        labeled_test_umap_features=data_container.data["labeled_test_umap_features"],
        labeled_test_and_unlabeled_train_umap_features=data_container.data["labeled_test_and_unlabeled_train_umap_features"],
        unlabeled_train_umap_features=data_container.data["unlabeled_train_umap_features"],
        cluster_preds=df_labeled_test_and_unlabeled_train["cluster"],
        title="HDBSCAN with cluster contours and test misclassifications",
        run=None,
        loop=None,
        save_path=save_path
    )


if __name__ == '__main__':
    main()
