import os

import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Union, Tuple, Dict, Any

from hil.utils.gallery_helper import save_gallery_images, read_text_from_gallery, get_label_and_comment
from hil.utils.base_gallery import Gallery
from experiment_setups.real_HiL.core.helper import create_folders, get_most_incorrect_class
from hil.utils.general_helper import create_directory
from hil.general_hil_logic import HumanInTheLoopBase


class Colors:
    RED = '\033[91m'
    GREEN = '\033[92m'
    DARK_YELLOW = '\033[38;5;130m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    END = '\033[0m'


class RealHiLLogic(HumanInTheLoopBase):
    def __init__(
            self,
            save_path: str,
            root_path: str,
            experiment_name: str,
            run: int,
            loops: List[Union[str, int]],
            random_seed: int,
            class_names: List[str],
            predictor_opt: Dict[str, Any],
            logger,
            data_processor,
            num_of_relabeled_cells: int,
            max_relabeling_pool_size: int,
            clustering_plot_in_each_loop: bool,
            shuffle_after_first_run: bool = False,
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
            data_processor=data_processor,
            process_unlabeled=True
        )

        self.root_path = root_path
        self.ctc_class = class_names["CTC"]
        self.non_ctc_class = class_names["Non-CTC"]
        self.num_of_relabeled_cells = num_of_relabeled_cells

        self.max_relabeled_pool_size = max_relabeling_pool_size
        self.relabeling_pool_idx_lst = []
        self.relabeled_cells_idx_lst = []
        self.new_labels = []

        self.train_pool_features = np.array([])
        self.relabeling_pool_features = np.array([])

        self.shuffle_after_first_run = shuffle_after_first_run

    def prepare_paths(self, current_loop: Union[int, str]) -> None:
        """
        Prepare directory paths for the current loop.
        """
        self.res_path = os.path.join(self.save_path, "init" if current_loop == "init" else f"loop_{current_loop}")
        if not os.path.exists(self.res_path):
            create_directory(self.res_path)

    def create_folders(self):
        """
        Create required folders for real hil experiments.
        """
        if not os.path.exists(os.path.join(self.res_path, "model")):
            create_folders(
                output_folder=self.res_path
            )

    def define_initial_relabeling_pool(self, current_loop: Union[int, str]) \
            -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
        """
        Define and return the initial relabeling pool data.
        It also creates folders for the current loop if they do not already exist.

        Params:
            current_loop (Union[int, str]): Identifier for the current loop of the Human-in-the-Loop run.
                                            "init" signifies the initialization stage, whereas integers denote
                                            ongoing sampling/relabeling loops.
        Returns:
            Tuple containing:
                - relabeling_pool_df (pd.DataFrame): DataFrame containing the sample info selected for relabeling.
                - relabeling_pool_pca_features (np.ndarray): PCA features of the relabeling pool samples.
                - relabeling_pool_labels (np.ndarray): The labels of the relabeling pool samples.
                - relabeling_pool_umap_features (np.ndarray): UMAP features of the relabeling pool samples.
        """
        self.prepare_paths(current_loop)
        self.create_folders()

        relabeling_pool_df = self.data.df_unlabeled_train
        relabeling_pool_pca_features = self.data.unlabeled_train_pca_features
        relabeling_pool_labels = self.data.train_unlabels
        relabeling_pool_umap_features = self.data.unlabeled_train_umap_features

        self.relabeling_pool_features = self.data.unlabeled_train_features

        return relabeling_pool_df, relabeling_pool_pca_features, relabeling_pool_labels, relabeling_pool_umap_features

    def define_initial_train_pool(self, current_loop: Union[int, str]) \
            -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
        """
        Define and return the initial training pool data.

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
        train_pool_df = self.data.df_labeled_train
        train_pool_pca_features = self.data.labeled_train_pca_features
        train_pool_labels = self.data.train_labels
        train_pool_umap_features = self.data.labeled_train_umap_features

        self.train_pool_features = self.data.train_features

        return train_pool_df, train_pool_pca_features, train_pool_labels, train_pool_umap_features

    def apply_sampling_strategy(self, current_loop):
        """
        Apply sample strategy for the current loop based on the experiment mode (cluster-specific or random).
        It also creates folders for the current loop if they do not already exist.

        Params:
            current_loop (Union[int, str]): Identifier for the current loop of the Human-in-the-Loop run.
                                            "init" signifies the initialization stage, while integers represent
                                            subsequent sampling/relabeling loops.

        """
        self.prepare_paths(current_loop)
        self.create_folders()

        if self.experiment_name == "cluster_specific":
            self._cluster_specific_sampling()
        if self.experiment_name == "random":
            self._random_sampling()

    def _cluster_specific_sampling(self):
        """
        Perform cluster-specific sampling:
        Samples are drawn from the cluster with the lowest F1 score and the most predicted incorrect class.
        """
        cluster_with_lowest_f1_score, most_incorrect_class = get_most_incorrect_class(
            test_pool_df=self.test_pool_df,
            df_cluster_metrics=self.df_cluster_metrics
        )

        # Select indices for relabeling within the specified cluster and class
        self.relabeling_pool_idx_lst = np.where(np.array(
            (self.relabeling_pool_df["cluster"] == cluster_with_lowest_f1_score) &
            (self.relabeling_pool_df["preds"] == most_incorrect_class)))[0]
        self.relabeling_pool_idx_lst = list(set(self.relabeling_pool_idx_lst))

        # Limit the size of the relabeling pool
        random.seed(self.random_seed)
        if len(self.relabeling_pool_idx_lst) > self.max_relabeled_pool_size:
            self.relabeling_pool_idx_lst = random.sample(self.relabeling_pool_idx_lst, self.max_relabeled_pool_size)
        else:
            # Shuffle relabeling pool indices
            random.Random(self.random_seed).shuffle(self.relabeling_pool_idx_lst)

    def _random_sampling(self):
        """
        Perform random sampling:
        Samples (most predicted incorrect class) are randomly drawn from the relabeling pool across all clusters.
        """

        _, most_incorrect_class = get_most_incorrect_class(
            test_pool_df=self.test_pool_df,
            df_cluster_metrics=self.df_cluster_metrics
        )

        # Select indices of incorrect predictions for random relabeling
        preds_idx_lst = np.where(np.array(self.relabeling_pool_df["preds"] == most_incorrect_class))[0]
        relabeling_pool_idx_lst = self.relabeling_pool_df.iloc[preds_idx_lst].index.to_list()
        self.relabeling_pool_idx_lst = list(set(relabeling_pool_idx_lst))

        # Limit size and shuffle the relabeling pool
        random.seed(self.random_seed)
        if len(self.relabeling_pool_idx_lst) > self.max_relabeled_pool_size:
            self.relabeling_pool_idx_lst = random.sample(self.relabeling_pool_idx_lst, self.max_relabeled_pool_size)
        else:
            # Shuffle relabeling pool indices
            random.Random(self.random_seed).shuffle(self.relabeling_pool_idx_lst)

    def _plot_relabel_gallery(self):

        """
        Plot the gallery of images for human relabeling.

        This method depends on:
            - `self.relabeling_pool_idx_lst`: List of indices to be used for relabeling.
            - `self.relabeling_pool_df`: Dataframe containing the data for relabeling images.
            - `self.res_path`: Path where the gallery images will be saved.
            - `self.root_path`: Root path for accessing the initial data.

        Save the gallery in the same folder as the main file. If a gallery already exists at the expected file path,
        it will not be recreated.
        """

        expected_file_path = os.path.join(self.res_path, "gallery", "relabel_pdf", "_gallery_view.pdf")
        if not os.path.exists(expected_file_path):
            dapi_img_path_lst = []
            ck_img_path_lst = []
            cd45_img_path_lst = []
            overlay_dapi_ck_img_path_lst = []

            # Save images for the relabeling gallery
            for i in range(len(self.relabeling_pool_idx_lst)):
                dapi_img_save_path, ck_img_save_path, cd45_img_save_path, overlay_img_save_path = save_gallery_images(
                    path=self.root_path,
                    df=self.relabeling_pool_df.iloc[self.relabeling_pool_idx_lst],
                    save_path=os.path.join(self.res_path, "gallery"),
                    index_counter=i
                )
                dapi_img_path_lst.append(dapi_img_save_path)
                ck_img_path_lst.append(ck_img_save_path)
                cd45_img_path_lst.append(cd45_img_save_path)
                overlay_dapi_ck_img_path_lst.append(overlay_img_save_path)

            # Plot and display the gallery for relabeling
            gallery = Gallery()
            gallery.plot_gallery(
                case="",
                proben_id="",
                df=self.relabeling_pool_df.iloc[self.relabeling_pool_idx_lst],
                overlay_img_path_lst=overlay_dapi_ck_img_path_lst,
                dapi_img_path_lst=dapi_img_path_lst,
                ck_img_path_lst=ck_img_path_lst,
                cd45_img_path_lst=cd45_img_path_lst,
                column_name_1=None,
                column_name_2=None
            )
            plt.close("all")

    def _relabel_by_human(self) -> Tuple[List[int], List[int]]:
        """
        Performs relabeling based on human input.

        This function attempts to open and read the gallery PDF from the expected location
        (`your_current_res_path/gallery/relabel_pdf/_gallery_view.pdf`). This is particularly useful when:
        a) repeating an experiment, or
        b) you wish to resume from where you left off previously.

        If the gallery file is not found at the expected path, you will be prompted to provide an alternative path.

        Instructions for relabeling the gallery:

        - If you have not yet relabeled the images in the gallery, first open the gallery file.
          Initially, the gallery file will be located in the same directory as the main file.

        - Add your labels (0 / 1) and, once you are finished, save the file. For saving, it is recommended to save
        the file at the expected file path without changing its name. If you choose to save the file in a different
        folder, you may or may not change the filename, but be aware that you will need to provide the alternative
        path each time you repeat the experiment.

        - If you need to end the session without completing the relabeling of the current gallery and do not want to
        lose your changes, you should either save the file under a different name (other than "_gallery_view.pdf")
        if saving in the current directory (where `main.py` is located) or the expected directory, or save it in a
        different location entirely. Reasons for this include:
        a) If you save the file with the same name in the expected directory, the function will read this file,
        preventing you from continuing the relabeling.
        b) If saved with the same name in the current directory, your changes will be overwritten by the initial
        version of the file, losing your annotations.

        Returns:
            Tuple[List[int], List[int]]: Indices of relabeled cells; corresponding new labels.
        """

        expected_file_path = os.path.join(self.res_path, "gallery", "relabel_pdf", "_gallery_view.pdf")

        try:
            df_relabel = self.load_relabel_data(expected_file_path)
            return self.process_relabel_data(df_relabel)
        except (FileNotFoundError, ValueError) as e:
            print(f"Error: {str(e)}")
            return [], []

    def load_relabel_data(self, expected_file_path: str) -> pd.DataFrame:
        """
        Load annotations from a relabeled gallery file.

        This function attempts to load the annotations from a gallery file located at the specified expected path.
        If the file cannot be found at the given location, the user is prompted to provide an alternative file path.

        Returns:
            pd.DataFrame: Annotations from the gallery as a dataframe.
            If the file cannot be located at the specified path, the system will request the correct path from the user.

        """
        try:
            return read_text_from_gallery(file_path=expected_file_path)
        except FileNotFoundError:
            return self.get_user_relabel_file(expected_file_path)

    def get_user_relabel_file(self, expected_file_path: str) -> pd.DataFrame:
        """
        Prompt the user to confirm the relabeling status and load the relabeled gallery file.

        Params:
            expected_file_path (str): The path where the gallery file is expected to be found. The expected file path
                                      will be printed to the console.

        Returns:
            pd.DataFrame: A DataFrame containing the annotations extracted from the gallery file.
        """

        print(f"{Colors.DARK_YELLOW}The gallery PDF file was not found in the expected folder.{Colors.END}\n")
        print(f"{Colors.DARK_YELLOW}Expected file path: {expected_file_path}{Colors.END}\n")
        print(f"{Colors.DARK_YELLOW}I created gallery file in the same directory as the main file.{Colors.END}\n")

        answer = input(f"{Colors.BLUE}Have you relabeled the images? "
                       "Please enter 1 or 2 for:\n"
                       "1): Yes, but it's in another folder.\n"
                       "2): No, I haven't relabeled yet but will do now/ I have to continue relabeling.\n"
                       f"If you haven't relabeled yet, you can find a newly created gallery file in the same "
                       f"directory as the main file. \n"
                       f"Important: Use only {self.ctc_class} (= CTC) and {self.non_ctc_class} (=Non-CTC) as labels in the gallery file: {Colors.END}. \n").strip()

        if answer == "1":
            return self.get_file_path_from_user()
        elif answer == "2":
            self.wait_for_user_to_complete_labeling()
            try:
                return read_text_from_gallery(file_path=expected_file_path)
            except FileNotFoundError:
                return self.get_file_path_from_user()
        else:
            raise ValueError(f"{Colors.RED}Invalid input. Continuing with empty lists.{Colors.END}")

    @staticmethod
    def get_file_path_from_user() -> pd.DataFrame:
        """
        Prompt the user to enter a file path until a valid file is found.
        """
        print(f"{Colors.GREEN}IMPORTANT: Do not save with the same name in the directory where it was created.\n")
        while True:
            alt_file_path = input(
                f"{Colors.GREEN}Enter the absolute path of the labeled gallery file:{Colors.END}\n").strip()
            try:
                return read_text_from_gallery(file_path=alt_file_path)
            except FileNotFoundError:
                print(f"{Colors.GREEN}File not found. Ensure you entered the correct path{Colors.END}.")

    @staticmethod
    def wait_for_user_to_complete_labeling():
        """
        Prompt the user to complete relabeling and save the file.
        """
        print(f"{Colors.GREEN}Please add your labels. Once you are finished, save it to the expected file path:")
        print(f"{Colors.GREEN}IMPORTANT: Do not save with the same name in the directory where it was created.\n")

        while True:
            done_confirmation = input(
                f"{Colors.GREEN}Type 'done' when you have completed labeling: {Colors.END}").strip().lower()
            if done_confirmation == "done":
                break

    def process_relabel_data(self, df_relabel: pd.DataFrame) -> Tuple[List[int], List[int]]:
        """
        Extract all labels and comments, add it to corresponding dataframe (df_preselection) and return an updated df.
        Indices with labels 0 and 1 are identified.

        Params:
            df_relabel (pd.Dataframe): Consists of the information extracted by the relabeled gallery.

        Returns:
            Tuple[List[int], List[int]]: Indices of relabeled cells, corresponding new labels.

        """
        df_added, _, _ = get_label_and_comment(
            df=df_relabel,
            df_preselection=self.relabeling_pool_df.iloc[self.relabeling_pool_idx_lst],
            select=False
        )

        ctc_index_lst = df_added.index[df_added['label'] == self.ctc_class].tolist()
        nonctc_index_lst = df_added.index[df_added['label'] == self.non_ctc_class].tolist()
        relabeled_cells_idx_lst = ctc_index_lst + nonctc_index_lst

        df_relabel = df_added.loc[relabeled_cells_idx_lst]
        new_labels = [int(i) for i in df_relabel["label"].to_list()]
        relabeled_cells_idx_lst = df_relabel["ref_index"].to_list()

        np.save(os.path.join(self.res_path, "data", "relabeled_cells_idx_lst.npy"), relabeled_cells_idx_lst)
        self.relabeled_cells_idx_lst = np.concatenate((self.relabeled_cells_idx_lst, relabeled_cells_idx_lst))
        self.relabeled_cells_idx_lst = [int(i) for i in self.relabeled_cells_idx_lst]
        self.num_of_relabeled_cells = np.concatenate((self.num_of_relabeled_cells, [len(relabeled_cells_idx_lst)]))
        self.num_of_relabeled_cells = [int(i) for i in self.num_of_relabeled_cells]
        self.new_labels = np.concatenate((self.new_labels, new_labels))
        self.new_labels = [int(i) for i in self.new_labels]

        return relabeled_cells_idx_lst, new_labels

    @staticmethod
    def get_sliced_indices(
            current_loop: Union[int, str],
            relabeled_cells_idx_lst: List[int],
            new_labels: List[int],
            num_of_relabeled_cells: List[int]
    ) -> Tuple[List[int], List[int]]:
        """
        Get sliced indices and corresponding new labels for the given loop iteration.
        """
        if not 1 <= current_loop <= len(num_of_relabeled_cells):
            raise ValueError(
                f"Invalid 'current_loop' value: {current_loop}. "
                f"It must be between 1 and {len(num_of_relabeled_cells)}."
            )

        num_start = sum(num_of_relabeled_cells[:current_loop - 1])
        num_end = num_start + num_of_relabeled_cells[current_loop - 1]

        return relabeled_cells_idx_lst[num_start:num_end], new_labels[num_start:num_end]

    def _shuffle_relabel_cells_idx_list(self, current_loop: Union[int, str]):
        """
        This function is required when no additional cells are identified for labeling after a complete hil run.
        Shuffle the list of relabeled cell indices for subsequent hil repetitions while maintaining
        the same number of new samples per loop.
        E.g. first hil run: [11, 10, 7, 3] (number of samples that are relabeled in e.g. loop: 1,2,3,4)
        Then in the next hil run, the number of new samples is maintained in each loop.
        """
        path = os.path.dirname(self.res_path.split("run_")[0])
        relabeled_cells_idx_lst = list(
            np.load(os.path.join(path, "run_1", "relabeled_cells_idx_lst_run_1.npy")))
        random.Random(self.random_seed).shuffle(relabeled_cells_idx_lst)
        copy_lst = list(np.load(os.path.join(path, "run_1", "relabeled_cells_idx_lst_run_1.npy")))
        corresponding_indices = [copy_lst.index(item) for item in relabeled_cells_idx_lst]
        new_labels = np.load(os.path.join(path, "run_1", "new_labels.npy"))
        new_labels = new_labels[corresponding_indices]

        relabeled_cells_idx_lst, new_labels = self.get_sliced_indices(
            current_loop,
            relabeled_cells_idx_lst,
            new_labels,
            self.num_of_relabeled_cells
        )

        return relabeled_cells_idx_lst, new_labels

    def get_indices_and_new_labels(self, current_loop: Union[int, str]) -> Tuple[List[int], List[int]]:
        if self.shuffle_after_first_run and self.run > 1:
            relabeled_cells_idx_lst, new_labels = self._shuffle_relabel_cells_idx_list(current_loop)
        else:
            self._plot_relabel_gallery()
            relabeled_cells_idx_lst, new_labels = self._relabel_by_human()

        mask = self.relabeling_pool_df['ref_index'].isin(relabeled_cells_idx_lst)
        relabeled_cells_idx_lst = self.relabeling_pool_df[mask].index.to_list()
        # add labels to df
        self.relabeling_pool_df.loc[relabeled_cells_idx_lst, "label"] = new_labels

        return relabeled_cells_idx_lst, new_labels

    def _get_data_for_cluster_plot(self):
        self.df_unlabeled_train = self.relabeling_pool_df
        self.df_labeled_test_and_unlabeled_train = pd.concat([self.test_pool_df, self.relabeling_pool_df],
                                                             ignore_index=True)
        self.labeled_test_and_unlabeled_train_umap_features = np.concatenate(
            (self.test_pool_umap_features, self.relabeling_pool_umap_features), axis=0)
        self.unlabeled_train_umap_features = self.relabeling_pool_umap_features

    def update_data(self, relabeled_cells_idx_lst, new_labels):
        super().update_data(relabeled_cells_idx_lst, new_labels)

        self.train_pool_features = np.concatenate((self.train_pool_features,
                                                   self.relabeling_pool_features[relabeled_cells_idx_lst]))
        self.relabeling_pool_features = np.delete(self.relabeling_pool_features, relabeled_cells_idx_lst, axis=0)

    def save_data(self):
        results_data_path = os.path.join(self.res_path, "data")

        self.train_pool_df.to_csv(os.path.join(results_data_path, "train_pool_df.csv"))
        self.relabeling_pool_df.to_csv(os.path.join(results_data_path, "relabeling_pool_df.csv"))
        self.test_pool_df.to_csv(os.path.join(results_data_path, "test_pool_df.csv"))

        np.save(os.path.join(results_data_path, "train_pool_features"), self.train_pool_features)
        np.save(os.path.join(results_data_path, "relabeling_pool_features"), self.relabeling_pool_features)

        np.save(os.path.join(results_data_path, "train_pool_labels"), self.train_pool_labels)
        np.save(os.path.join(results_data_path, "relabeling_pool_labels"), self.relabeling_pool_labels)

        np.save(os.path.join(results_data_path, "train_pool_pca_features"), self.train_pool_pca_features)
        np.save(os.path.join(results_data_path, "relabeling_pool_pca_features"), self.relabeling_pool_pca_features)

        np.save(os.path.join(results_data_path, "train_pool_umap_features"), self.train_pool_umap_features)
        np.save(os.path.join(results_data_path, "relabeling_pool_umap_features"), self.relabeling_pool_umap_features)
