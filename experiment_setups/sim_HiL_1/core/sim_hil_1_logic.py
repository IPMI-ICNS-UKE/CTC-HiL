import os
import random

import numpy as np
import pandas as pd
from typing import List, Union, Tuple, Dict, Any

from experiment_setups.sim_HiL_1.core.helper import create_simulation_experiment_folders, \
    monte_carlo_cross_validation_of_training_pool, \
    general_metrics_of_monte_carlo_results
from hil.utils.general_helper import cluster_metrics, create_directory
from hil.general_hil_logic import HumanInTheLoopBase


class SimHiL1Logic(HumanInTheLoopBase):
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
            num_of_relabeled_cells: int,
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

        self.num_of_relabeled_cells = num_of_relabeled_cells

        self.relabeling_pool_idx_lst = []
        self.relabeled_cells_idx_lst = []
        self.new_labels = []
        self.num_samples_per_cluster = []
        self.random_samples = []

        self.df_labeled_train_selected = pd.DataFrame()
        self.train_labels_selected = List
        self.labeled_train_pca_features_selected = np.array([])
        self.labeled_train_umap_features_selected = np.array([])

        self.current_df_mc_general_metrics = pd.DataFrame()
        self.current_df_mc_cluster_metrics = pd.DataFrame()
        self.df_mc_general_metrics = pd.DataFrame()
        self.df_mc_cluster_metrics = pd.DataFrame()

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

    def define_initial_relabeling_pool(self, current_loop: Union[int, str]) \
            -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
        """
        Define and return the initial relabeling pool data.
        """
        self.prepare_paths(current_loop)
        self.create_folders()

        random.seed(self.random_seed)
        selected_indices = random.sample(self.data.df_labeled_train.index.to_list(), self.num_of_relabeled_cells)

        self.df_labeled_train_selected = self.data.df_labeled_train.iloc[selected_indices].reset_index(drop=True)
        self.train_labels_selected = self.data.df_labeled_train['label'].iloc[selected_indices].to_numpy()
        self.labeled_train_pca_features_selected = self.data.labeled_train_pca_features[selected_indices]
        self.labeled_train_umap_features_selected = self.data.labeled_train_umap_features[selected_indices]

        # Remove the selected samples from the original labeled training set
        relabeling_pool_df = self.data.df_labeled_train.drop(index=selected_indices).reset_index(drop=True)
        relabeling_pool_labels = relabeling_pool_df["label"].to_numpy()
        relabeling_pool_pca_features = np.delete(self.data.labeled_train_pca_features, selected_indices, axis=0)
        relabeling_pool_umap_features = np.delete(self.data.labeled_train_umap_features, selected_indices, axis=0)

        return relabeling_pool_df, relabeling_pool_pca_features, relabeling_pool_labels, relabeling_pool_umap_features

    def define_initial_train_pool(self, current_loop: Union[int, str]) \
            -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
        """
        Define and return the initial training pool data.
        """
        # train_pool_df = self.df_labeled_train_selected
        # train_pool_pca_features = self.labeled_train_pca_features_selected
        # train_pool_labels = self.train_labels_selected
        # train_pool_umap_features = self.labeled_train_umap_features_selected

        return (self.df_labeled_train_selected, self.labeled_train_pca_features_selected, self.train_labels_selected,
                self.labeled_train_umap_features_selected)

    def fit_classifier_and_give_general_metrics(self, current_loop: Union[int, str]) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Get general metric results and perform monte carlo cross validation.
        """
        general_metrics_result = super().fit_classifier_and_give_general_metrics(current_loop)
        self.monte_carlo_cross_validation(current_loop, self.train_pool_labels,
                                          self.train_pool_pca_features, self.train_pool_df)
        return general_metrics_result

    def monte_carlo_cross_validation(
            self,
            current_loop: Union[int, str],
            train_pool_labels: np.ndarray,
            train_pool_pca_features: np.ndarray,
            train_pool_df: pd.DataFrame
    ) -> None:
        """
        Perform Monte Carlo cross-validation and update summary metrics.
        """
        df_mc = monte_carlo_cross_validation_of_training_pool(
            classifier_opt=self.predictor_opt,
            train_labels=train_pool_labels,
            labeled_train_pca_features=train_pool_pca_features,
            df=train_pool_df,
            model_save_path=os.path.join(self.res_path, "cv")
        )

        self.current_df_mc_general_metrics = general_metrics_of_monte_carlo_results(
            experiment_name=self.experiment_name + str("_cv_mc"),
            run=self.run,
            loop=current_loop,
            df_mc=df_mc,
        )

        self.current_df_mc_cluster_metrics = cluster_metrics(
            experiment_name=self.experiment_name + str("_cv_mc"),
            cluster_lst=np.unique(df_mc["cluster"]),
            cluster_preds=df_mc["cluster"],
            labels=df_mc["label"],
            preds=df_mc["preds"],
            run=self.run,
            loop=current_loop,
        )

        self.df_mc_general_metrics = pd.concat(
            [self.df_mc_general_metrics, self.current_df_mc_general_metrics], ignore_index=True)
        self.df_mc_cluster_metrics = pd.concat(
            [self.df_mc_cluster_metrics, self.current_df_mc_cluster_metrics], ignore_index=True)

    def apply_sampling_strategy(self, current_loop: Union[int, str]) -> None:
        """
        Suggest samples for the current loop based on the experiment mode (cluster-specific or random).
        """
        self.prepare_paths(current_loop)
        self.create_folders()

        if self.experiment_name == "cluster_specific":
            self.num_samples_per_cluster = self._cluster_specific_sampling()

        if self.experiment_name == "random":
            self.random_samples = self._random_sampling()

    def _cluster_specific_sampling(self) -> List[int]:
        """
        Perform cluster-specific sampling: calculates for each cluster the number of samples
        """
        mc_f1_scores = self.current_df_mc_cluster_metrics["cluster f1"].to_list()
        # Sum of all cluster's f1 score
        sum_f1_score = sum((1 - cluster) for cluster in mc_f1_scores)
        num_samples_per_cluster = []

        for cluster in range(len(self.current_df_mc_cluster_metrics)):
            mc_f1_score = mc_f1_scores[cluster]
            cluster_f1_score = (1 - mc_f1_score)
            frequency = (cluster_f1_score / sum_f1_score)
            num_samples = int(np.round(frequency * self.num_of_relabeled_cells))
            num_samples_per_cluster.append(num_samples)
        sum_samples = sum(num_samples_per_cluster)

        if sum_samples != self.num_of_relabeled_cells:
            # E.g. because of rounding
            difference = self.num_of_relabeled_cells - sum_samples
            # update num of samples of cluster with lowest f1 score
            lowest_score_idx = mc_f1_scores.index(min(mc_f1_scores))
            num_samples_per_cluster[lowest_score_idx] += difference

        return num_samples_per_cluster

    def _random_sampling(self):
        """
        Perform random sampling.
        """
        random.seed(self.random_seed)
        random_samples = random.sample(self.relabeling_pool_df.index.to_list(), self.num_of_relabeled_cells)

        return random_samples

    def get_indices_and_new_labels(self, current_loop: Union[int, str]) -> Tuple[List[int], List[int]]:
        """
        Get indices of relabeled cells and corresponding labels after cluster-specific or random sampling.
        """
        relabeled_cells_idx_lst = None
        new_labels = None
        if self.experiment_name == "cluster_specific":
            relabeled_cells_idx_lst, new_labels = self.cluster_sample_selector()
        if self.experiment_name == "random":
            relabeled_cells_idx_lst = self.random_samples
            new_labels = self.relabeling_pool_df.iloc[relabeled_cells_idx_lst]["label"].to_list()
        return relabeled_cells_idx_lst, new_labels

    def cluster_sample_selector(self) -> Tuple[List[int], List[int]]:
        """
        Check if each cluster has enough samples based on calculated number of samples per cluster (cluster-specific sampling).
        If not, samples are drawn from the next cluster with the smallest F1 score, ensuring no duplicates.
        Returns indices of samples and corresponding labels.

        """
        mc_f1_scores = self.current_df_mc_cluster_metrics["cluster f1"].tolist()
        cluster_names = self.current_df_mc_cluster_metrics["cluster name"].tolist()

        sample_indices_per_cluster = []

        for num, cluster in enumerate(cluster_names):
            random.seed(self.random_seed)

            # Get cluster indices from df
            idx_lst = self.get_cluster_indices(cluster, sample_indices_per_cluster)

            # Extract sufficient number of samples
            idx_lst = self.extract_samples_from_cluster(idx_lst, num, mc_f1_scores, cluster_names)
            sample_indices_per_cluster.extend(idx_lst)

        new_labels = self.get_labels_for_indices(sample_indices_per_cluster)
        return sample_indices_per_cluster, new_labels

    def get_cluster_indices(self, cluster: int, sample_indices_per_cluster: List) -> List[int]:
        """
        Retrieve indices of all samples belonging to the given cluster.
        It ensures that indices which are already present in `sample_indices_per_cluster` are excluded in idx_lst.
        This is useful for preventing the re-selection of samples that have already been processed.
        """
        idx_lst = self.relabeling_pool_df[self.relabeling_pool_df['cluster'] == cluster].index.tolist()
        idx_lst = [idx for idx in idx_lst if idx not in sample_indices_per_cluster]
        return idx_lst

    def extract_samples_from_cluster(self, idx_lst: List[int], num: int, mc_f1_scores: List[float],
                                     cluster_names: List[int]) -> List[int]:
        """Attempt to extract the required number of samples from a given cluster."""
        if len(idx_lst) >= self.num_samples_per_cluster[num]:
            return random.sample(idx_lst, self.num_samples_per_cluster[num])

        additional_samples = list(idx_lst)
        num_of_samples_to_draw = self.num_samples_per_cluster[num] - len(idx_lst)

        while num_of_samples_to_draw > 0:
            next_cluster, mc_f1_scores = self.get_next_cluster_with_lowest_f1(mc_f1_scores, cluster_names)
            additional_sample_idx_lst = self.get_cluster_indices(next_cluster, cluster_names)

            # Draw as many samples as needed or available
            drawn_samples = random.sample(additional_sample_idx_lst,
                                          min(num_of_samples_to_draw, len(additional_sample_idx_lst)))
            additional_samples.extend(drawn_samples)
            num_of_samples_to_draw -= len(drawn_samples)

        return additional_samples

    def get_next_cluster_with_lowest_f1(self, mc_f1_scores: List[float], cluster_names: List[int]) -> Tuple[
        int, List[float]]:
        """Find the next cluster with the lowest F1 score and update the score list."""
        lowest_score = min(mc_f1_scores)
        lowest_score_index = mc_f1_scores.index(lowest_score)
        mc_f1_scores.pop(lowest_score_index)
        cluster_names_copy = cluster_names.copy()
        cluster_names_copy.pop(lowest_score_index)
        return cluster_names_copy[lowest_score_index], mc_f1_scores

    def get_labels_for_indices(self, indices: List[int]) -> List[int]:
        """Retrieve labels for a given list of indices."""
        return self.relabeling_pool_df.loc[indices, "label"].tolist()

    def _get_data_for_cluster_plot(self) -> None:
        """
        Prepare data for cluster plot.
        """
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
