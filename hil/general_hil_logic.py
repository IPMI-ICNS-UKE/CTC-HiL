import os
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from typing import List, Union, Tuple, Dict, Any

from hil.utils.general_helper import general_metrics, cluster_metrics
from hil.utils.cluster_plot import plot_cluster_contours_in_hdbscan


class HumanInTheLoopBase(ABC):
    """
    Base class for Human-in-the-Loop (hil) logic to improve classification performance by relabeling data.
    """

    def __init__(
            self,
            save_path: str,
            experiment_name: str,
            run: int,
            loops: List[Union[str, int]],
            random_seed: int,
            predictor_opt: Dict[str, Any],
            logger,
            clustering_plot_in_each_loop,
            data_processor=None,
            data_container=None,
            process_unlabeled=False,
    ):

        self.loops = loops
        self.run = run
        self.predictor_opt = predictor_opt
        self.logger = logger
        self.save_path = save_path
        self.experiment_name = experiment_name
        self.random_seed = random_seed
        self.clustering_plot_in_each_loop = clustering_plot_in_each_loop
        self.process_unlabeled = process_unlabeled

        # Choose the correct data source
        self.data = data_processor if data_processor else data_container

        # Initialize dataframes
        self.df_general_metrics = pd.DataFrame()
        self.df_cluster_metrics = pd.DataFrame()

        # data needed for cluster plot
        self.df_unlabeled_train = pd.DataFrame()
        self.df_labeled_test_and_unlabeled_train = pd.DataFrame()
        self.unlabeled_train_umap_features = np.array([])
        self.labeled_test_and_unlabeled_train_umap_features = np.array([])

        # Initialize cluster predictions
        self.cluster_pred = np.array([])
        self.cluster_pred_unlab_train = np.array([])

        # States and relabeling pools
        self.cluster_metrics_dict = {}

        self.train_pool_df = pd.DataFrame()
        self.train_pool_pca_features = np.array([])
        self.train_pool_labels = np.ndarray
        self.train_pool_umap_features = np.array([])

        self.test_pool_df = pd.DataFrame()
        self.test_pool_pca_features = np.array([])
        self.test_pool_labels = np.ndarray
        self.test_pool_umap_features = np.array([])

        self.relabeling_pool_df = pd.DataFrame()
        self.relabeling_pool_pca_features = np.array([])
        self.relabeling_pool_labels = np.ndarray
        self.relabeling_pool_umap_features = np.array([])

        self.res_path = str()

    @abstractmethod
    def define_initial_relabeling_pool(self, current_loop: Union[int, str])\
            -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
        """
        Establishes the initial relabeling pool for a given loop (we call it "current_loop")
        in the current (meaning: ongoing) hil run.

        This method should be implemented in subclasses to define how the initial set of samples for relabeling
        is determined.

        Params:
            current_loop (Union[int, str]): Identifier for the current loop of the Human-in-the-Loop run.
                                            It can be a string representing the initialization as "init" or an integer
                                            representing the sampling/ relabeling loop.

        Returns:
            Tuple containing:
                - relabeling_pool_df (pd.DataFrame): DataFrame containing the sample info selected for relabeling.
                - relabeling_pool_pca_features (np.ndarray): PCA features of the relabeling pool samples.
                - relabeling_pool_labels (np.ndarray): The labels of the relabeling pool samples.
                - relabeling_pool_umap_features (np.ndarray): UMAP features of the relabeling pool samples.
            """
        raise NotImplementedError("This method should be overridden in the subclass.")

    @abstractmethod
    def define_initial_train_pool(self, current_loop: Union[int, str]) \
            -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
        """
        Establishes the initial training pool for a given loop (we call it "current_loop")
        in the current (meaning: ongoing) hil run.

        This method should be implemented in subclasses to define how the initial set of samples for training
        is determined.

        Params:
            current_loop (Union[int, str]): Identifier for the current loop of the Human-in-the-Loop run.
                                            It can be a string representing the initialization as "init" or an integer
                                            representing the sampling/ relabeling loop.

        Returns:
            Tuple containing:
                - train_pool_df (pd.DataFrame): DataFrame containing the sample info selected for training.
                - train_pool_pca_features (np.ndarray): PCA features of the training pool samples.
                - train_pool_labels (np.ndarray): The labels of the training pool samples.
                - train_pool_umap_features (np.ndarray): UMAP features of the training pool samples.
            """
        raise NotImplementedError("This method should be overridden in the subclass.")

    def define_test_pool_data(self):
        """
        Initializes the test data pool for evaluation in the Human-in-the-Loop run.

        This method sets up the necessary data structures required for evaluating the classifier's performance.

        Preconditions:
            Assumes that the data container `self.data` has been initialized and contains the required
            DataFrame and feature arrays (df_labeled_test, labeled_test_pca_features,
            labeled_test_umap_features) before calling this method.
        """

        self.test_pool_df = self.data.df_labeled_test
        self.test_pool_pca_features = self.data.labeled_test_pca_features
        self.test_pool_labels = self.data.df_labeled_test["label"].to_numpy()
        self.test_pool_umap_features = self.data.labeled_test_umap_features

    @abstractmethod
    def apply_sampling_strategy(self, current_loop: Union[int, str]):
        """
        Define and execute the sampling strategy for selecting data points to be relabeled.

        This method should be implemented in subclasses to apply the sampling strategy
        tailored to the needs of the experiment (cluster_specific or random). The strategy determines how samples
        should be selected from the available data pool for relabeling.

        Params:
            current_loop (Union[int, str]): Identifier for the current loop of the Human-in-the-Loop run.
                                            "init" signifies the initialization stage, while integers represent
                                            subsequent sampling/relabeling loops.

        """
        raise NotImplementedError("This method should be overridden in the subclass.")

    @abstractmethod
    def get_indices_and_new_labels(self, current_loop: Union[int, str]) -> Tuple[List[int], List[int]]:
        """
        This function should return the indices of relabeled samples and their new labels for the current loop.

        This method should be implemented in subclasses to identify which samples have been relabeled
        during the current loop and to determine their updated labels. It plays a critical in the
        Human-in-the-Loop process by enabling iterative improvement of the model through updated data labels.

        Params:
            current_loop (Union[int, str]): Identifier for the current loop of the Human-in-the-Loop run.
                                            "init" signifies the initialization stage, while integers represent
                                            subsequent sampling/relabeling loops.

        Returns:
            Tuple containing:
                - List[int]: A list of indices of the selected (relabeled) samples.
                - List[int]: A list of new labels corresponding to the indices of the relabeled data.
        """
        raise NotImplementedError("This method should be overridden in the subclass.")

    def update_data(self, relabeled_cells_idx_lst: List[int], new_labels: List[int]):
        """
        Update training and relabeling datasets with newly relabeled samples.

        This function transfers samples from the relabeling pool to the training pool based on their indices
        and updates their labels.

        Params:
            relabeled_cells_idx_lst (List[int]): A list of indices identifying the samples within the relabeling pool
                                                 that have been relabeled.
            new_labels (List[int]): The new labels assigned to the samples specified by the indices in the
                                    relabeled_cells_idx_lst.
        """

        self.train_pool_df = pd.concat([
            self.train_pool_df, self.relabeling_pool_df.iloc[relabeled_cells_idx_lst]]
        ).reset_index(drop=True)
        self.relabeling_pool_df = self.relabeling_pool_df.drop(index=relabeled_cells_idx_lst).reset_index(
            drop=True)

        self.train_pool_labels = np.concatenate((self.train_pool_labels, new_labels))
        self.relabeling_pool_labels = np.delete(self.relabeling_pool_labels, relabeled_cells_idx_lst, axis=0)

        self.train_pool_pca_features = np.concatenate((self.train_pool_pca_features,
                                                       self.relabeling_pool_pca_features[relabeled_cells_idx_lst]))
        self.relabeling_pool_pca_features = np.delete(self.relabeling_pool_pca_features, relabeled_cells_idx_lst,
                                                      axis=0)

        self.train_pool_umap_features = np.concatenate((self.train_pool_umap_features,
                                                        self.relabeling_pool_umap_features[
                                                            relabeled_cells_idx_lst]))
        self.relabeling_pool_umap_features = np.delete(self.relabeling_pool_umap_features, relabeled_cells_idx_lst,
                                                       axis=0)

    @abstractmethod
    def save_data(self):
        """
        Save the current state of data pools.
        """
        raise NotImplementedError("This method should be overridden in the subclass.")

    def fit_classifier_and_give_general_metrics(
            self,
            current_loop: Union[int, str]
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Fit a classifier on the current training data and compute general performance metrics.

        Params:
            current_loop (Union[int, str]): Identifier for the current loop of the Human-in-the-Loop run.
                                            "init" signifies the initialization stage, whereas integers denote
                                            ongoing sampling/relabeling loops.
        Returns:
            Tuple:
                - pd.DataFrame: A DataFrame containing general metrics about the model's performance, such as
                                accuracy, F1 score, recall, and precision.
                - np.ndarray: An array of predicted labels for the test dataset.

        """
        # Additional arguments if needed
        if self.process_unlabeled:
            extra_args = {
                "unlabeled_train_set": self.relabeling_pool_pca_features,
                "train_unlabels": self.relabeling_pool_labels,
                "df_unlabeled_train": self.relabeling_pool_df,
            }
        else:
            extra_args = {
                "unlabeled_train_set": None,
                "train_unlabels": None,
                "df_unlabeled_train": None,
            }

        # General_metrics function
        result = general_metrics(
            experiment_name=self.experiment_name,
            opt=self.predictor_opt,
            train_set=self.train_pool_pca_features,
            test_set=self.test_pool_pca_features,
            train_labels=self.train_pool_labels,
            test_labels=self.test_pool_labels,
            process_train_unlabeled_data=self.process_unlabeled,
            model_save_path=os.path.join(self.res_path, "model"),
            run=self.run,
            loop=current_loop,
            **extra_args
        )

        return result

    def give_cluster_metrics(
            self,
            current_loop: Union[int, str],
    ) -> pd.DataFrame:
        """
        Calculate and return cluster-specific performance metrics for the current loop.

        Params:
            current_loop (Union[int, str]): Identifier for the current loop of the Human-in-the-Loop run.
                                            "init" signifies the initialization stage, whereas integers denote
                                            ongoing sampling/relabeling loops.
        Returns:
            pd.DataFrame: A DataFrame containing cluster-specific metrics, such as cluster accuracy, precision,
                          recall, F1 scores for the current loop.
        """
        df_cluster_metrics = cluster_metrics(
            experiment_name=self.experiment_name,
            cluster_lst=np.unique(self.test_pool_df["cluster"]),
            cluster_preds=self.test_pool_df["cluster"],
            labels=self.test_pool_labels,
            preds=self.test_pool_df["preds"],
            run=self.run,
            loop=current_loop,
        )

        return df_cluster_metrics

    @abstractmethod
    def _get_data_for_cluster_plot(self):
        """
        Prepare and assemble data necessary for generating cluster plots.
        """
        raise NotImplementedError("This method should be overridden in the subclass.")

    def human_in_the_loop(self):

        for loop in self.loops:
            current_loop = loop

            if current_loop == "init":
                (self.relabeling_pool_df, self.relabeling_pool_pca_features, self.relabeling_pool_labels,
                 self.relabeling_pool_umap_features) = self.define_initial_relabeling_pool(current_loop)

                (self.train_pool_df, self.train_pool_pca_features, self.train_pool_labels,
                 self.train_pool_umap_features) = self.define_initial_train_pool(current_loop)

                self.define_test_pool_data()

            if current_loop != "init":
                self.apply_sampling_strategy(current_loop)
                relabeled_cells_idx_lst, new_labels = self.get_indices_and_new_labels(current_loop)
                self.update_data(relabeled_cells_idx_lst, new_labels)
                self.save_data()

            general_metrics_result = self.fit_classifier_and_give_general_metrics(current_loop)
            if self.process_unlabeled and current_loop == "init":
                df_general_metrics, test_preds, self.relabeling_pool_df = general_metrics_result
                self.df_general_metrics = pd.concat([self.df_general_metrics, df_general_metrics], ignore_index=True)

            else:
                df_general_metrics, test_preds = general_metrics_result
                self.df_general_metrics = pd.concat([self.df_general_metrics, df_general_metrics], ignore_index=True)

            self.process_unlabeled = False
            self.test_pool_df["preds"] = test_preds
            df_cluster_metrics = self.give_cluster_metrics(current_loop)
            self.df_cluster_metrics = pd.concat([self.df_cluster_metrics, df_cluster_metrics], ignore_index=True)

            if self.clustering_plot_in_each_loop:
                self._get_data_for_cluster_plot()
                plot_cluster_contours_in_hdbscan(
                    train_unlabels=self.df_unlabeled_train["label"],
                    test_labels=self.test_pool_df["label"],
                    test_preds=self.test_pool_df["preds"],
                    labeled_test_umap_features=self.test_pool_umap_features,
                    labeled_test_and_unlabeled_train_umap_features=self.labeled_test_and_unlabeled_train_umap_features,
                    unlabeled_train_umap_features=self.unlabeled_train_umap_features,
                    cluster_preds=self.df_labeled_test_and_unlabeled_train["cluster"],
                    title="HDBSCAN with cluster contours and test misclassifications",
                    run=self.run,
                    loop=current_loop,
                    save_path=self.res_path
                )
