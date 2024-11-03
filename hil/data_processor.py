import os
import pickle

import pandas as pd
import numpy as np
from typing import List
from torch import nn
import umap
from sklearn.decomposition import PCA
import hdbscan

from hil.utils.general_helper import read_dataframe, load_features
from hil.cluster_identificator import ClusterIdentificator


class DataProcessor:
    """
    DataProcessor for handling the loading, transformation, and processing
    of datasets for labeled and unlabeled images.
    """

    def __init__(self,
                 root_path: str,
                 data_save_path: str,
                 example_path_to_df_of_train_case: str,
                 example_path_to_df_of_test_case: str,
                 train_cases: List,
                 test_cases: List,
                 device: int,
                 batch_size: int,
                 num_workers: int,
                 model: nn.Module,
                 pca: PCA,
                 umap_: umap.UMAP,
                 clusterer: hdbscan.HDBSCAN,
                 df_unlabeled_train: pd.DataFrame,
                 unlabeled_train_features: np.ndarray,
                 train_unlabels: np.ndarray
                 ):
        """
        Args:
        df_unlabeled_train (pd.DataFrame): DataFrame for the unlabeled training data.
        unlabeled_train_features (np.ndarray): Features for the unlabeled training data.
        train_unlabels (list): Labels for the training data.
        """
        self.root_path = root_path
        self.data_save_path = data_save_path
        self.example_path_to_df_of_train_case = example_path_to_df_of_train_case
        self.example_path_to_df_of_test_case = example_path_to_df_of_test_case

        self.train_cases = train_cases
        self.test_cases = test_cases

        self.device = device
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.model = model

        self.pca = pca
        self.umap_ = umap_
        self.clusterer = clusterer

        self.df_labeled_train = pd.DataFrame()
        self.df_labeled_test = pd.DataFrame()
        self.df_unlabeled_train = df_unlabeled_train
        self.df_labeled_test_and_unlabeled_train = pd.DataFrame()

        self.train_img_paths = np.array([])
        self.test_img_paths = np.array([])

        self.train_labels = np.array([])
        self.test_labels = np.array([])
        self.train_unlabels = train_unlabels

        self.train_features = np.array([])
        self.test_features = np.array([])
        self.unlabeled_train_features = unlabeled_train_features

        self.labeled_train_pca_features = np.array([])
        self.labeled_test_pca_features = np.array([])
        self.unlabeled_train_pca_features = np.array([])

        self.labeled_train_umap_features = np.array([])
        self.labeled_test_umap_features = np.array([])
        self.unlabeled_train_umap_features = np.array([])
        self.labeled_test_and_unlabeled_train_umap_features = np.array([])

        self.cluster_identificator = ClusterIdentificator()

        self.read_dataframes()
        self.concatenate_df()
        self.save_df()
        self.get_features()
        self.pca_transformation()
        self.umap_transforamtion()
        self.clustering()

    def read_dataframes(self):
        """
        Load dataframes for labeled train and test datasets.
        """

        # Load labeled training dataset
        self.df_labeled_train = read_dataframe(
            case_lst=self.train_cases,
            path_to_df=self.example_path_to_df_of_train_case,
            datatype="labeled",
        )

        # Load labeled test dataset
        self.df_labeled_test = read_dataframe(
            case_lst=self.test_cases,
            path_to_df=self.example_path_to_df_of_test_case,
            datatype="labeled",
        )

    def concatenate_df(self):
        self.df_labeled_test_and_unlabeled_train = pd.concat([self.df_labeled_test, self.df_unlabeled_train])

    def save_df(self):
        self.df_labeled_train.to_csv(os.path.join(self.data_save_path, "df_labeled_train", ), index=False)
        self.df_labeled_test.to_csv(os.path.join(self.data_save_path, "df_labeled_test"), index=False)
        self.df_unlabeled_train.to_csv(os.path.join(self.data_save_path, "df_unlabeled_train"), index=False)
        self.df_labeled_test_and_unlabeled_train.to_csv(os.path.join(
            self.data_save_path, "df_labeled_test_and_unlabeled_train"), index=False)

    def get_features(self):
        """
        Extract and save features for train and test datasets.
        """
        # Extract image paths
        self.train_img_paths = self.df_labeled_train.loc[:, ["dapi_path", "ck_path", "cd45_path"]].values.tolist()
        self.test_img_paths = self.df_labeled_test.loc[:, ["dapi_path", "ck_path", "cd45_path"]].values.tolist()

        # Convert labels to appropriate format
        train_labels = self.df_labeled_train["label"].astype('int64').to_list()
        test_labels = self.df_labeled_test["label"].to_list()

        # Load features for train dataset
        self.train_features, self.train_labels = load_features(
            root_path=self.root_path,
            img_paths=self.train_img_paths,
            label_list=train_labels,
            model=self.model,
            device=self.device,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            save_path=self.data_save_path,
            features_file_name="train_features",
            labels_file_name="train_labels"
        )

        # Load features for test dataset
        self.test_features, self.test_labels = load_features(
            root_path=self.root_path,
            img_paths=self.test_img_paths,
            label_list=test_labels,
            model=self.model,
            device=self.device,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            save_path=self.data_save_path,
            features_file_name="test_features",
            labels_file_name="test_labels"
        )

    def pca_transformation(self):
        """
        Apply PCA to reduce dimensionality of features and save/load transformed features.
        """
        try:
            # Attempt to load precomputed PCA features
            self.labeled_train_pca_features = np.load(
                os.path.join(self.data_save_path, "labeled_train_pca_features.npy"),
            )
            self.labeled_test_pca_features = np.load(
                os.path.join(self.data_save_path, "labeled_test_pca_features.npy")
            )
            self.unlabeled_train_pca_features = np.load(
                os.path.join(self.data_save_path, "unlabeled_train_pca_features.npy")
            )
        except FileNotFoundError:
            # Fit PCA and save PCA features if not found
            self.labeled_train_pca_features = self.pca.fit_transform(self.train_features)
            self.labeled_test_pca_features = self.pca.transform(self.test_features)
            self.unlabeled_train_pca_features = self.pca.transform(self.unlabeled_train_features)

            # Save
            np.save(os.path.join(
                self.data_save_path, "labeled_train_pca_features.npy"), self.labeled_train_pca_features
            )
            np.save(os.path.join(
                self.data_save_path, "labeled_test_pca_features.npy"), self.labeled_test_pca_features
            )
            np.save(os.path.join(
                self.data_save_path, "unlabeled_train_pca_features.npy"), self.unlabeled_train_pca_features
            )

    def umap_transforamtion(self):
        """
        Apply UMAP for further dimensionality reduction and visualization and save the results.
        """
        try:
            umap_transformation = np.load(
                os.path.join(self.data_save_path, "UMAP_transformation_on_lab_train_features_pca.sav"), allow_pickle=True
            )
            self.labeled_train_umap_features = umap_transformation.embedding_
            self.labeled_test_umap_features = np.load(
                os.path.join(self.data_save_path, "labeled_test_umap_features.npy")
            )
            self.unlabeled_train_umap_features = np.load(
                os.path.join(self.data_save_path, "unlabeled_train_umap_features.npy")
            )
            self.labeled_test_and_unlabeled_train_umap_features = np.load(
                os.path.join(self.data_save_path, "labeled_test_and_unlabeled_train_umap_features.npy")
            )
            print(f"Loaded umap arrays from files.")

        except FileNotFoundError:
            # Fit UMAP if precomputed arrays not found
            self.umap_.fit(self.labeled_train_pca_features)
            self.labeled_train_umap_features = self.umap_.embedding_
            self.labeled_test_umap_features = self.umap_.transform(self.labeled_test_pca_features)
            self.unlabeled_train_umap_features = self.umap_.transform(self.unlabeled_train_pca_features)
            self.labeled_test_and_unlabeled_train_umap_features = np.concatenate(
                (self.labeled_test_umap_features, self.unlabeled_train_umap_features), axis=0)

            # Save
            umap_file = os.path.join(self.data_save_path, "UMAP_transformation_on_lab_train_features_pca.sav")
            pickle.dump(self.umap_, open(umap_file, 'wb'))
            np.save(os.path.join(self.data_save_path, "labeled_train_umap_features.npy"),
                    self.labeled_train_umap_features)
            np.save(os.path.join(self.data_save_path, "labeled_test_umap_features.npy"),
                    self.labeled_test_umap_features)
            np.save(os.path.join(self.data_save_path, "unlabeled_train_umap_features.npy"),
                    self.unlabeled_train_umap_features)
            np.save(os.path.join(self.data_save_path, "labeled_test_and_unlabeled_train_umap_features.npy"),
                    self.labeled_test_and_unlabeled_train_umap_features)

    def clustering(self):
        # Perform the latent space cluster analysis:
        # Clustering in the two-dimensional space is carried out using
        # Hierarchical Density-Based Spatial Clustering of Applications with Noise (HDBSCAN)

        cluster_preds = self.cluster_identificator.find_clusters(
            umap_features=self.labeled_test_and_unlabeled_train_umap_features,
            clusterer=self.clusterer
        )

        self.df_labeled_test, self.df_unlabeled_train, _ = self.cluster_identificator.assign_clusters(
            cluster_pred=cluster_preds,
            df_labeled_test=self.df_labeled_test,
            df_unlabeled_train=self.df_unlabeled_train,
        )