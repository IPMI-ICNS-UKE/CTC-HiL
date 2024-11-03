import os
import random

import numpy as np
import pandas as pd
from typing import List, Union, Tuple, Dict, Any

from experiment_setups.sim_HiL_2.core.helper import create_simulation_experiment_folders
from experiment_setups.sim_HiL_2.core.helper import split_randomly_by_percentage
from hil.utils.general_helper import create_directory
from hil.general_hil_logic import HumanInTheLoopBase


class SimHiL2Logic(HumanInTheLoopBase):
    def __init__(
            self,
            save_path: str,
            experiment_name: str,
            run: int,
            loops: List[Union[str, int]],
            random_seed: int,
            predictor_opt: Dict[str, Any],
            logger,
            data_container,
            main_cluster: int,
            remaining_clusters: List[int],
            clustering_plot_in_each_loop: bool,
    ):

        super().__init__(
            save_path,
            experiment_name,
            run,
            loops,
            random_seed,
            predictor_opt,
            logger,
            clustering_plot_in_each_loop,
            data_processor=data_container,
            process_unlabeled=False
        )

        self.main_cluster = main_cluster
        self.remaining_clusters = remaining_clusters

        self.main_cluster_random_relabling_pool_indices = []

        # for initial training pool
        self.train_sample_indices = []

        self.num_samples_per_loop = []
        self.sample_indices = []

    def prepare_paths(self, current_loop: Union[int, str]) -> None:
        """
        Prepare directory paths for the current loop.
        """
        self.res_path = os.path.join(self.save_path, "init" if current_loop == "init" else f"loop_{current_loop}")
        if not os.path.exists(self.res_path):
            create_directory(self.res_path)

    def create_folders(self) -> None:
        """
                Create required folders for simulation experiments.
                """
        if not os.path.exists(os.path.join(self.res_path, "model")):
            create_simulation_experiment_folders(
                output_folder=self.res_path
            )

    def split_cluster_data(self, cluster: int, percentage: float) -> Tuple[List[int], List[int]]:
        """
        Splits data based on a percentage for a given cluster.
        """
        return split_randomly_by_percentage(
            lst=self.data.df_labeled_train["cluster"],
            seed=self.random_seed,
            cluster=cluster,
            percentage=percentage
        )

    def calculate_pool_sizes(self, main_cluster: int, remaining_clusters: List[int]) \
            -> Tuple[int, int, List[int], List[int], List[int]]:
        """
        Calculates the cluster-specific and random relabeling pool sizes.
        """
        # Indices of main cluster: 80% and 20%
        main_cluster_80, main_cluster_20 = self.split_cluster_data(main_cluster, 0.8)
        num_20_percent_samples = len(main_cluster_20)
        cluster_specific_relabeling_pool_size = len(main_cluster_80)

        # Indices of the remaining clusters, divided into 80% and 20%
        remaining_clusters_80 = []
        remaining_clusters_20 = []
        for remaining_cluster in remaining_clusters:
            remaining_cluster_80, remaining_cluster_20 = self.split_cluster_data(remaining_cluster, 0.8)
            remaining_clusters_80.extend(remaining_cluster_80)
            remaining_clusters_20.extend(remaining_cluster_20)

        random_relabeling_pool_size = num_20_percent_samples + len(remaining_clusters_20)

        if self.experiment_name == "random":
            random.seed(self.random_seed)
            # we use 20% of the left-out labeled data of the main cluster
            self.main_cluster_random_relabling_pool_indices = random.sample(main_cluster_80, num_20_percent_samples)

        # the training data set of only a single cluster (referred to as the main cluster) is here pruned to
        # 20% of its original size, while all others are limited to 80% of their original size.
        self.train_sample_indices = main_cluster_20 + remaining_clusters_80

        return cluster_specific_relabeling_pool_size, random_relabeling_pool_size, main_cluster_80, main_cluster_20, \
            remaining_clusters_20

    @staticmethod
    def adjust_relabeling_pool(main_cluster_80, new_relabeling_pool_size) -> List[int]:
        """
        Adjusts the relabeling pool to the new size if the relabeling pool from the cluster-specific strategy exceeds
        that of the random approach.
        """
        random.seed(42)
        main_cluster_new_80 = random.sample(main_cluster_80, new_relabeling_pool_size)
        return main_cluster_new_80

    def calculate_samples_per_loop(self, pool_size: int, num_of_20_percent_samples: int) -> List[int]:
        """
        Calculates the number of samples per loop.
        """
        num_samples_per_loop = []
        for loop in range(len(self.loops)):
            if loop == len(self.loops) - 1:  # Last loop
                last_20_percent_samples = pool_size - sum(num_samples_per_loop[1:])
                num_samples_per_loop.append(last_20_percent_samples)
            else:
                num_samples_per_loop.append(num_of_20_percent_samples)

        return num_samples_per_loop

    def define_initial_relabeling_pool(self, current_loop: Union[int, str]) \
            -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:

        self.prepare_paths(current_loop)
        self.create_folders()

        cluster_specific_relabeling_pool_size, random_relabeling_pool_size, main_cluster_80, main_cluster_20, \
            remaining_clusters_20 = self.calculate_pool_sizes(self.main_cluster, self.remaining_clusters)

        num_samples_20 = len(main_cluster_20)

        if cluster_specific_relabeling_pool_size > random_relabeling_pool_size:
            main_cluster_80 = self.adjust_relabeling_pool(main_cluster_80, random_relabeling_pool_size)
            cluster_specific_relabeling_pool_size = len(main_cluster_80)
            num_samples_20 = int(np.round(cluster_specific_relabeling_pool_size / len(self.loops[1:])))

        self.num_samples_per_loop = self.calculate_samples_per_loop(cluster_specific_relabeling_pool_size,
                                                                    num_samples_20)
        indices = None
        if self.experiment_name == "cluster_specific":
            # indices of 80% percent of the main cluster
            indices = main_cluster_80
        if self.experiment_name == "random":
            # The relabeling pool consists of:
            # 20% of the left-out labeled data of the main cluster &
            # 20% of the left-out labeled data of each of the remaining clusters
            indices = self.main_cluster_random_relabling_pool_indices + remaining_clusters_20

        relabeling_pool_df = self.data.df_labeled_train.iloc[indices].reset_index(drop=True)
        relabeling_pool_pca_features = self.data.labeled_train_pca_features[indices]
        relabeling_pool_labels = self.data.df_labeled_train['label'].iloc[indices].to_numpy()
        relabeling_pool_umap_features = self.data.labeled_train_umap_features[indices]

        return relabeling_pool_df, relabeling_pool_pca_features, relabeling_pool_labels, relabeling_pool_umap_features

    def define_initial_train_pool(self, current_loop: Union[int, str]) \
            -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:

        train_pool_df = self.data.df_labeled_train.iloc[self.train_sample_indices].reset_index(drop=True)
        train_pool_pca_features = self.data.labeled_train_pca_features[self.train_sample_indices]
        train_pool_labels = self.data.df_labeled_train['label'].iloc[self.train_sample_indices].to_numpy()
        train_pool_umap_features = self.data.labeled_train_umap_features[self.train_sample_indices]

        return train_pool_df, train_pool_pca_features, train_pool_labels, train_pool_umap_features

    def apply_sampling_strategy(self, current_loop: Union[int, str]):

        self.prepare_paths(current_loop)
        self.create_folders()

        # The training and relabeling pools have been previously defined based on the chosen strategy,
        # allowing for random sampling from the relabeling pool.
        random.seed(self.random_seed)
        self.sample_indices = random.sample(self.relabeling_pool_df.index.to_list(),
                                            self.num_samples_per_loop[current_loop])

    def get_indices_and_new_labels(self, current_loop: Union[int, str]) -> Tuple[List[int], List[int]]:
        new_labels = self.relabeling_pool_df.iloc[self.sample_indices]["label"].to_list()

        return self.sample_indices, new_labels

    def _get_data_for_cluster_plot(self):
        self.df_unlabeled_train = self.data.df_unlabeled_train
        self.df_labeled_test_and_unlabeled_train = pd.concat([self.test_pool_df, self.data.df_unlabeled_train],
                                                             ignore_index=True)
        self.labeled_test_and_unlabeled_train_umap_features = self.data.labeled_test_and_unlabeled_train_umap_features
        self.unlabeled_train_umap_features = self.data.unlabeled_train_umap_features

    def save_data(self) -> None:
        results_data_path = os.path.join(self.res_path, "data")

        self.train_pool_df.to_csv(os.path.join(results_data_path, "train_pool_df.csv"))
        self.relabeling_pool_df.to_csv(os.path.join(results_data_path, "relabeling_pool_df.csv"))
        self.test_pool_df.to_csv(os.path.join(results_data_path, "test_pool_df.csv"))

        np.save(os.path.join(results_data_path, "train_pool_labels"), self.train_pool_labels)
        np.save(os.path.join(results_data_path, "relabeling_pool_labels"), self.relabeling_pool_labels)

        np.save(os.path.join(results_data_path, "train_pool_pca_features"), self.train_pool_pca_features)
        np.save(os.path.join(results_data_path, "relabeling_pool_pca_features"), self.relabeling_pool_pca_features)

        np.save(os.path.join(results_data_path, "train_pool_umap_features"), self.train_pool_umap_features)
        np.save(os.path.join(results_data_path, "relabeling_pool_umap_features"), self.relabeling_pool_umap_features)
