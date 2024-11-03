import os
import random
import time

import numpy as np
import pandas as pd
from alive_progress import alive_bar
from pprint import pformat

from experiment_setups.sim_HiL_2.core.sim_hil_2_logic import SimHiL2Logic
from hil.utils.logger import Logger
from hil.data_container import DataContainer
from hil.utils.general_helper import load_config, configure_paths, create_directory, save_results, \
    plot_results, assign_clusters


def main():
    """
    Main function to execute the simulation hil 2 experiment (cluster-specific or random).
    This function configures paths, initializes data, performs simulations,
    and saves as well as plots the results.

    In the context of multiple hil runs, where each run comprises an 'Initialization' phase followed by a series
    of sampling/ relabeling 'loops', we designate the ongoing run as 'current_run'. Similarly, within each hil run,
    we refer to the active loop as 'current_loop'.
    """

    configure_paths()

    # Load experiment configuration from config file
    cfg_path = "configs/sim_hil_2_cfg.yml"
    config = load_config(cfg_path)

    # Extract experiment configurations
    n_runs = config["HiL_configurations"]["n_runs"]  # Number of "hil runs"
    n_loops = config["HiL_configurations"][
        "n_loops"]  # "Initialization" and number of sampling & relabeling loops within a hil run
    experiment_name = config["experiment_setting"][
        "experiment_name"]  # 2 strategies to choose from: cluster_specific or random
    main_cluster = config["experiment_setting"]["main_cluster"]

    # Prepare directories for saving results
    save_path = os.path.join(config["general_paths"]["path_to_results_folder"], "sim_hil_2_results", experiment_name,
                             f"main_cluster_{main_cluster}")
    create_directory(save_path)
    plot_path = os.path.join(save_path, "plots")
    create_directory(plot_path)

    # Setup logger
    log_file_path = os.path.join(save_path, "simulation_HiL_1_logfile.log")
    logger = Logger(log_file_path=log_file_path)
    logger.info(f"Logger configured to write to {log_file_path}.")
    logger.info(f"Directories created at {save_path}.")
    logger.debug(f"Initializing Simulated hil 2 experiment with config: {pformat(config)}")

    # Set random seed for reproducibility
    random.seed(42)
    random_seed_lst = random.sample(range(1, 200), n_runs)

    # Load data from DataContainer
    data_container = DataContainer(
        path_to_input_folder=config["general_paths"]["path_to_data_folder"]
    )

    data_container.load_data()

    logger.info("Data loaded into DataContainer.")

    # Assign training data points to nearest clusters based on provided UMAP features.
    cluster_lst = assign_clusters(
        labeled_test_and_unlabeled_train_umap_features=data_container.labeled_test_and_unlabeled_train_umap_features,
        df_labeled_test_and_unlabeled_train=data_container.df_labeled_test_and_unlabeled_train,
        labeled_train_umap_features=data_container.labeled_train_umap_features,
        assign_background_cluster_no_nearest_cluster=False
    )

    data_container.add_column_to_df_labeled_train(new_column_data=cluster_lst, column_name="cluster")

    # Initialize DataFrames to store results
    df_general_results = pd.DataFrame()
    df_cluster_results = pd.DataFrame()

    with alive_bar(n_runs, title='Processing runs', force_tty=True, bar='blocks', stats=True) as bar:
        for run in range(n_runs):
            time.sleep(.001)
            current_run = run + 1
            random_seed = random_seed_lst[run]
            run_path = os.path.join(save_path, f"run_{current_run}")

            logger.info(f"Starting run {current_run} with random seed {random_seed} for main cluster {main_cluster}.")

            unique_cluster_names = np.unique(cluster_lst).tolist()
            remaining_clusters = [cluster for cluster in unique_cluster_names if cluster != main_cluster]

            # Initialize and execute simulation logic 2
            logger.info(f"Initializing SimHiL2Logic for experiment: {experiment_name}.")
            simulation_2 = SimHiL2Logic(
                save_path=run_path,
                experiment_name=experiment_name,
                run=current_run,
                loops=n_loops,
                random_seed=random_seed,
                predictor_opt=config["OPT"],
                logger=logger,
                data_container=data_container,
                main_cluster=main_cluster,
                remaining_clusters=remaining_clusters,
                clustering_plot_in_each_loop=config["HiL_configurations"]["clustering_plot_in_each_loop"],
            )

            logger.debug("Human in the loop.")
            simulation_2.human_in_the_loop()

            df_general_metrics = simulation_2.df_general_metrics
            df_general_results = pd.concat([df_general_results, df_general_metrics],
                                           ignore_index=True)

            df_cluster_metrics = simulation_2.df_cluster_metrics
            df_cluster_results = pd.concat([df_cluster_results, df_cluster_metrics],
                                           ignore_index=True)

            bar()

    df_cluster_results_for_plotting = save_results(
        n_runs=n_runs,
        save_path=save_path,
        experiment_name=experiment_name,
        df_general_results=df_general_results,
        df_cluster_results=df_cluster_results
    )

    # Line plot
    plot_results(
        experiment_name=experiment_name,
        df_cluster_results_for_plotting=df_cluster_results_for_plotting,
        save_path=save_path,
    )


if __name__ == '__main__':
    main()
