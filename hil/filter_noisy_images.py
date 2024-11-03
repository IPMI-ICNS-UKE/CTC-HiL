import os

from typing import List
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import sklearn as sk

from hil.utils.general_helper import read_dataframe, load_features, map_unique_elements_with_counts, define_and_fit_classifier


class NoisyClassifier:
    """
    A classifier to filter out noisy images from an unlabeled training
    set, leveraging a smaller labeled subset to train the model. The images
    are classified into 'noisy' (label: -1) and 'not noisy' (label: 2 -> meaning: it's either class 0 (Non-CTC) or
    1 (CTC)) categories. Only the 'not noisy' images are retained.
    """

    def __init__(self,
                 root_path: str,
                 data_save_path: str,
                 path_to_df_train_noisy_classifier: str,
                 example_path_to_df_of_train_case: str,
                 train_cases: List,
                 device: int,
                 batch_size: int,
                 num_workers: int,
                 model,
                 pca: PCA,
                 predictor_opt,
                 pos_label: int
    ):
        """
        Initialize the Noisy Classifier with configuration settings.

        Args:
        config_dict (dict): Contains paths, device settings, and other required configurations.
        """
        self.root_path = root_path
        self.data_save_path = data_save_path
        self.example_path_to_df_of_train_case = example_path_to_df_of_train_case
        self.path_to_df_train_noisy_classifier = path_to_df_train_noisy_classifier
        self.train_cases = train_cases
        self.device = device
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.predictor_opt = predictor_opt
        self.pos_label = pos_label

        self.model = model
        self.pca = pca

        self.train_img_paths = np.array([])
        self.train_labels = np.array([])
        self.train_features = np.array([])
        self.train_pca_features = np.array([])
        self.train_preds = np.array([])

        self.df_train_unlabeled = pd.DataFrame()
        self.train_unlabeled_img_paths = np.array([])
        self.train_unlabels = np.array([])
        self.train_unlabeled_features = np.array([])
        self.train_unlabeled_pca_features = np.array([])

        # Define the training set and extract features
        self.define_train_set()
        self.get_train_features()

    def define_train_set(self):
        """
        Prepare the train set for the classifier
        """

        # Reads dataframe of train data for the Noisy Classifier
        # The dataframe should have the followng columns
        #   - "label": image label
        #   - "dapi_path": path to the DAPI image
        #   - "ck_path": path to the CK image
        #   - "cd45_path": path to the CD45 image
        df_train_noisy_classifier = pd.read_csv(self.path_to_df_train_noisy_classifier)

        # Extract image paths and labels from the labeled data
        self.train_img_paths = df_train_noisy_classifier.loc[:, ["dapi_path", "ck_path", "cd45_path"]].values.tolist()
        self.train_labels = df_train_noisy_classifier["label"].astype('int64').to_list()

        # Combined the dataframe of each provided case into one
        df_train_unlabeled = read_dataframe(
            case_lst=self.train_cases,
            path_to_df=self.example_path_to_df_of_train_case,
            datatype="unlabeled",
        )
        df_train_unlabeled["ref_index"] = df_train_unlabeled.index.to_list()

        # Update unlabeled train set
        df_train_unlabeled = df_train_unlabeled.drop(index=df_train_noisy_classifier.ref_index.to_list())
        self.df_train_unlabeled = df_train_unlabeled.reset_index(drop=True)

        # Extract image paths from the unlabeled data
        self.train_unlabeled_img_paths = df_train_unlabeled.loc[
                                         :, ["dapi_path", "ck_path", "cd45_path"]
                                         ].values.tolist()
        # Label of unlabeled train is "10" -> meaning: it could either be a 1 (CTC) or 0 (Non-CTC)
        self.train_unlabels = df_train_unlabeled["label"].astype('int64').to_list()

    def get_train_features(self):
        """
        Extract features from images using the specified model. Apply PCA
        for dimensionality reduction if PCA features are not already cached.
        """

        # Extract features from the labeled images
        self.train_features, self.train_labels = load_features(
            root_path=self.root_path,
            img_paths=self.train_img_paths,
            label_list=self.train_labels,
            model=self.model,
            device=self.device,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            save_path=self.data_save_path,
            features_file_name="train_features_noisy_classifier",
            labels_file_name="train_labels_noisy_classifier"
        )

        # Extract features from the unlabeled images
        self.train_unlabeled_features, self.train_unlabels = load_features(
            root_path=self.root_path,
            img_paths=self.train_unlabeled_img_paths,
            label_list=self.train_unlabels,
            model=self.model,
            device=self.device,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            save_path=self.data_save_path,
            features_file_name="train_unlabeled_features_noisy_classifier",
            labels_file_name="train_unlabels_noisy_classifier"
        )

        # Attempt to load existing PCA-transformed features; if unavailable, compute them
        try:
            self.train_pca_features = np.load(
                os.path.join(self.data_save_path, "train_features_pca_noisy_classifier.npy")
            )
            self.train_unlabeled_pca_features = np.load(
                os.path.join(self.data_save_path, "train_unlabeled_features_pca_noisy_classifier.npy")
            )
            print("Loaded PCA features")
        except FileNotFoundError:
            print("Performing PCA")
            self.train_pca_features = self.pca.fit_transform(self.train_features)
            self.train_unlabeled_pca_features = self.pca.transform(self.train_unlabeled_features)
            np.save(
                os.path.join(self.data_save_path, "train_features_pca_noisy_classifier.npy"),
                self.train_pca_features
            )
            np.save(
                os.path.join(self.data_save_path, "train_unlabeled_features_pca_noisy_classifier.npy"),
                self.train_unlabeled_pca_features
            )

    def sort_out_noisy_images(self):
        """
        Use the trained classifier to sort out 'noisy' images and retain 'not noisy' ones.
        Returns:
        (pd.DataFrame, np.ndarray, np.ndarray): Returns a DataFrame of filtered
        unlabeled images and their respective features and labels.
        """
        predictor = define_and_fit_classifier(
            opt=self.predictor_opt,
            features=self.train_pca_features,
            labels=self.train_labels,
            save_path=os.path.join(self.data_save_path),
            file_name="noisy_classifier_model"
        )

        # Unlabeled train predictions
        train_unlabeled_preds = predictor.predict(self.train_unlabeled_pca_features)
        print(f"Unlabeled train predictions: ", map_unique_elements_with_counts(train_unlabeled_preds))

        self.df_train_unlabeled["preds_sel"] = train_unlabeled_preds
        self.df_train_unlabeled = self.df_train_unlabeled.loc[self.df_train_unlabeled["preds_sel"] == self.pos_label]
        train_unlabeled_features = self.train_unlabeled_features[self.df_train_unlabeled.index.to_list()]
        train_unlabels = self.train_unlabels[self.df_train_unlabeled.index.to_list()]
        df_train_unlabeled = self.df_train_unlabeled.reset_index(drop=True)

        return df_train_unlabeled, train_unlabeled_features, train_unlabels
