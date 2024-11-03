import os
import random
import time

import numpy as np
import pandas as pd
from tqdm import tqdm
from pprint import pformat
from sklearn.decomposition import PCA
import umap
import hdbscan
from alive_progress import alive_bar

from experiment_setups.real_HiL.core.real_hil_logic import RealHiLLogic
from hil.utils.logger import Logger
from hil.utils.general_helper import save_results, plot_results, load_config, configure_paths, create_directory, load_model
from hil.utils.model_zoo import xcit_gc_nano_12_p8_224_dist
from hil.filter_noisy_images import NoisyClassifier
from hil.data_processor import DataProcessor


def main():
    """
    Main function to perform the real Human-in-the-loop experiment (cluster_specific or random).

    In the context of multiple hil runs, where each run comprises an 'Initialization' phase followed by a series
    of sampling/ relabeling 'loops', we designate the ongoing run as 'current_run'. Similarly, within each hil run,
    we refer to the active loop as 'current_loop'.

    This experiment is interactive in nature. During each loop (except for the initialization), a gallery in 
    PDF format, similar to the CellSearch gallery, will be generated in the same folder as this main Python script. 
    The program will then pause and ask you to assign labels. After labeling, instructions on how to proceed will 
    be displayed in the terminal. If you stop the relabeling process, please ensure that you provide the same 
    results path in the corresponding configuration file. This will allow you to resume from the exact point 
    where you left off.
    """

    configure_paths()

    # Load experiment configuration from config file
    cfg_path = "configs/real_hil_example_cfg.yml"
    config = load_config(cfg_path)

    # Extract experiment configurations
    n_runs = config["HiL_configurations"]["n_runs"]  # Number of "hil runs"
    n_loops = config["HiL_configurations"]["n_loops"]  # "Initialization" and number of sampling & relabeling loops within a hil run
    experiment_name = config["experiment_setting"]["experiment_name"]  # 2 strategies to choose from: cluster_specific or random

    # This is the Backbone we used for the DINO implementation
    model = xcit_gc_nano_12_p8_224_dist(in_chans=3, num_classes=0)
    model = load_model(model=model, device=config['device'], path=config['general_paths']['state_dict_path'])

    # Initialize PCA, UMAP and HDBSCAN
    pca = PCA(**config["PCA_parameters"])
    umap_ = umap.UMAP(**config["UMAP_parameters"])
    clusterer = hdbscan.HDBSCAN(**config["HDBSCAN_parameters"])

    # Prepare directories for saving results
    save_path = os.path.join(config["general_paths"]["path_to_results_folder"], experiment_name)
    create_directory(save_path)
    create_directory(os.path.join(save_path, "plots"))
    data_save_path = os.path.join(config["general_paths"]["path_to_results_folder"], "data")
    create_directory(data_save_path)

    # Setup logger
    log_file_path = os.path.join(save_path, "real_HiL_logfile.log")
    logger = Logger(log_file_path=log_file_path)
    logger.info(f"Logger configured to write to {log_file_path}.")
    logger.info(f"Directories created at {save_path}.")
    logger.debug(f"Initializing real hil experiment with config: {pformat(config)}")

    # Set random seed for reproducibility
    random.seed(42)
    random_seed_lst = random.sample(range(1, 200), n_runs)

    ########################################################################################
    # OPTIONAL:
    # The unlabeled train set consists of artefacts such as smeared cells and noisy data
    # Therefore: a simple classifier sorts out noisy images
    # If your data contains noisy data as well, you can follow the example of this class or construct your own.
    # If not, you can skip this part. But make sure, you provide the Dataprocessor a df for the unlabeled train set,
    # the corresponding features and labels

    noisy_classifier = NoisyClassifier(
        root_path=config['general_paths']['root_path'],
        data_save_path=data_save_path,
        path_to_df_train_noisy_classifier=config['general_paths']['path_to_df_train_noisy_classifier'],
        example_path_to_df_of_train_case=config['general_paths']['example_path_to_df_of_train_case'],
        train_cases=config['train_cases'],
        device=config['device'],
        batch_size=config["dataloader_parameters"]["batch_size"],
        num_workers=config["dataloader_parameters"]["num_workers"],
        model=model,
        pca=pca,
        predictor_opt=config["OPT"],
        pos_label=config["noisy_classifier"]["pos_label"]
    )

    # train unlabels means labels for the unlabeled train set.
    df_unlabeled_train, unlabeled_train_features, train_unlabels = noisy_classifier.sort_out_noisy_images()
    ########################################################################################

    # Class that processes the data (Dataloader, extracts features, creates PCA and UMAP features, clustering)
    data_processor = DataProcessor(
        root_path=config['general_paths']['root_path'],
        data_save_path=data_save_path,
        example_path_to_df_of_train_case=config['general_paths']['example_path_to_df_of_train_case'],
        example_path_to_df_of_test_case=config['general_paths']['example_path_to_df_of_test_case'],
        train_cases=config['train_cases'],
        test_cases=config['test_cases'],
        device=config['device'],
        batch_size=config["dataloader_parameters"]["batch_size"],
        num_workers=config["dataloader_parameters"]["num_workers"],
        model=model,
        pca=pca,
        umap_=umap_,
        clusterer=clusterer,
        df_unlabeled_train=df_unlabeled_train,
        unlabeled_train_features=unlabeled_train_features,
        train_unlabels=train_unlabels
    )

    # Dataframes to save the results
    df_general_results = pd.DataFrame()
    df_cluster_results = pd.DataFrame()

    num_of_relabeled_cells = []

    # Start with the hil runs
    for run in tqdm(range(n_runs)):
        current_run = run + 1
        random_seed = random_seed_lst[run]
        run_path = os.path.join(save_path, f"run_{current_run}")
        create_directory(run_path)

        # Perform the real-world hil experiment
        real_hil = RealHiLLogic(
            save_path=run_path,
            root_path=config['general_paths']['root_path'],
            experiment_name=experiment_name,
            run=current_run,
            loops=n_loops,
            random_seed=random_seed,
            class_names=config["HiL_configurations"]["class_names"],
            predictor_opt=config["OPT"],
            logger=logger,
            data_processor=data_processor,
            num_of_relabeled_cells=num_of_relabeled_cells,
            max_relabeling_pool_size=config["HiL_configurations"]["max_relabeling_pool_size"],
            clustering_plot_in_each_loop=config["HiL_configurations"]["clustering_plot_in_each_loop"],
            shuffle_after_first_run=config["HiL_configurations"]["shuffle_after_one_HiL_run"]
        )

        # Call the human-in-the-loop method
        real_hil.human_in_the_loop()

        ########################################################################################
        # OPTIONAL:
        # As described in the paper, after 1 hil run with the cluster-specific strategy, no cells are left to be
        # annotated. Therefore, we can store the number of new labeled samples for each loop in this list.
        # The new labeled samples are then shuffled in the remaining repeated hil runs,
        # while maintaining the same number of new samples per loop.
        # If you need this option, set the parameter shuffle_after_first_run in RealHiLLogic to True
        if experiment_name == "cluster_specific" and current_run == 1:
            relabeled_cells_idx_lst = real_hil.relabeled_cells_idx_lst
            np.save(os.path.join(run_path, "relabeled_cells_idx_lst_run_1.npy"), relabeled_cells_idx_lst)
            new_labels = real_hil.new_labels
            np.save(os.path.join(run_path, "new_labels.npy"), new_labels)
            num_of_relabeled_cells = real_hil.num_of_relabeled_cells
        ########################################################################################

        # Stores results of the current experiment in dataframes
        df_general_metrics = real_hil.df_general_metrics
        df_general_results = pd.concat([df_general_results, df_general_metrics], ignore_index=True)

        df_cluster_metrics = real_hil.df_cluster_metrics
        df_cluster_results = pd.concat([df_cluster_results, df_cluster_metrics], ignore_index=True)


    # Saves general and cluster results to Excel files
    df_cluster_results_for_plotting = save_results(
        n_runs=n_runs,
        save_path=save_path,
        experiment_name=experiment_name,
        df_general_results=df_general_results,
        df_cluster_results=df_cluster_results
    )

    # Creates the Line plot
    plot_results(
        experiment_name=experiment_name,
        df_cluster_results_for_plotting=df_cluster_results_for_plotting,
        save_path=save_path,
    )


if __name__ == '__main__':
    main()
