import os

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from alphashape import alphashape
from matplotlib.colors import to_hex
from typing import Union, Optional
import matplotlib.lines as mlines

from hil.utils.line_plot import prepare_contour_colors_and_labels

# Initial mapping and predefined colors can be extended if more clusters are found
INITIAL_LABEL_MAPPING = {
    '10.0': 'Unlabeled train data',
    '0.0': 'Non-CTC (test data)',
    '1.0': 'CTC (test data)',
    '-1': 'Background'
}

PREDEFINED_COLORS_PALETTE = ['#dfc181', '#ce92c8', 'g', '#95b282', '#828090']
PREDEFINED_AREA_PALETTE = ['#f6eedd', '#f2d1ef', '#e3f3e5', '#c2d0b6', '#cfcce5']


def normalize_label(label) -> str:
    """
    Format label strings.
    """
    return f"{float(label):.1f}" if '.' not in str(label) else str(label)


def plot_cluster_contours_in_hdbscan(
        train_unlabels: Union[pd.Series, np.ndarray],
        test_labels: Union[pd.Series, np.ndarray],
        test_preds: Union[pd.Series, np.ndarray],
        labeled_test_umap_features: np.ndarray,
        labeled_test_and_unlabeled_train_umap_features: np.ndarray,
        unlabeled_train_umap_features: np.ndarray,
        cluster_preds: Union[pd.Series, np.ndarray],
        title: str,
        run: Optional[int],
        loop: Optional[Union[str, int]],
        save_path: str,
        fig_size=(15, 15)
) -> None:

    """
    Plot HDBSCAN cluster contours.
    General idea is to plot 3 scatter plots and the contours of each cluster (except for the backgorund cluster)
    in just one plot in the following order:
    1. Scatter plot: Complete unlabeled train data (small points, e.g. gray)
    2. Scatter plot: Correct classified test data (small points, blue-ish). Plotted on top of scatter plot 1.
    3. Contours of clusters
    4. Scatter plot: Misclassified test data (symbol: cross; different colors for the predicted classes)

    Params:
        train_unlabels (np.ndarray): Labels of unlabeled train data (e.g. "10" for "it could either be a CTC (label: 1)
                                     or a non-CTC (label:0).
        test_labels (np.ndarray): True labels of test data.
        test_preds (np.ndarray): Predictions of labeled test data.
        labeled_test_umap_features (np.ndarray): UMAP features for labeled test data.
        labeled_test_and_unlabeled_train_umap_features (np.ndarray): Combined UMAP features for test and unlabeled train.
        unlabeled_train_umap_features (np.ndarray): UMAP features of unlabeled train data.
        cluster_preds (np.ndarray): Cluster predictions of combined labeled test data and unlabeled train data.
        title (str): Title for the plot.
        run (Optional[int]): Current run number, if applicable. "Current" refers to the hil run we're in right now
                             since we can perform multiple runs (each run has n number of loops).
        loop (Optional[Union[str, int]]): Current loop number, if applicable. The current loop can be the
                                          "initialization" or loop 1, 2, .. etc.)
        save_path (str): Path to save the plot.
        fig_size (Tuple[int, int]): Size of plot figure.
    """

    fig, ax = plt.subplots(figsize=fig_size)

    # Colors
    color_unlabeled_train = {10.0: (0.725, 0.78, 0.78)}
    colors_labeled_test = {0.0: (0.231, 0.651, 0.627), 1.0: (0.231, 0.651, 0.627)}

    # Index for correctly and incorrectly classified instances
    misclassified_idx_lst = np.where(test_labels != test_preds)[0].tolist()
    misclassifier_labeled_test_umap_features = labeled_test_umap_features[misclassified_idx_lst]
    test_misclassifications = test_preds[misclassified_idx_lst]
    correct_classified_idx_lst = np.where(test_labels == test_preds)[0].tolist()
    correct_classified_test_umap_features = labeled_test_umap_features[correct_classified_idx_lst]
    correct_test_preds = test_preds[correct_classified_idx_lst]

    # Scatter plot 1
    ax = sns.scatterplot(
        x=unlabeled_train_umap_features[:, 0],
        y=unlabeled_train_umap_features[:, 1],
        hue=train_unlabels,
        palette=color_unlabeled_train,
        linewidth=0.2,
        s=10,
        legend="full",
        alpha=0.8
    )

    # Scatter plot 2
    ax1 = sns.scatterplot(
        x=correct_classified_test_umap_features[:, 0],
        y=correct_classified_test_umap_features[:, 1],
        hue=correct_test_preds,
        palette=colors_labeled_test,
        linewidth=0.2,
        s=10,
        legend="full",
        alpha=0.8
    )

    ax.set_facecolor('none')

    # Prepare contours
    cluster_names = np.unique(cluster_preds).tolist()
    contours_color_palette, contours_color_area_palette, label_mapping = prepare_contour_colors_and_labels(
        cluster_names, INITIAL_LABEL_MAPPING, PREDEFINED_COLORS_PALETTE, PREDEFINED_AREA_PALETTE)

    counter = 0
    for cluster_name in cluster_names:
        if cluster_name != -1:
            current_cluster_idx_lst = cluster_preds[cluster_preds == cluster_name].index.to_list()
            # current_cluster_idx_lst = df_current_cluster.index.to_list()
            x = labeled_test_and_unlabeled_train_umap_features[:, 0][current_cluster_idx_lst]
            y = labeled_test_and_unlabeled_train_umap_features[:, 1][current_cluster_idx_lst]

            points = np.column_stack((x, y))

            outermost_points = None  # Initialize variable to store the result

            for alpha in range(5, 0, -1):
                try:
                    alpha_shape = alphashape(points, alpha)
                    if not alpha_shape.is_empty:
                        # Try to access the exterior, assuming the shape is a Polygon
                        outermost_points = np.array(alpha_shape.exterior.coords)
                        break  # Exit the loop if successful
                except AttributeError:
                    # If an AttributeError occurs (e.g., MultiPolygon with no .exterior), continue to the next alpha
                    continue

            if outermost_points is not None:
                # Plot contours
                plt.plot(outermost_points[:, 0],
                         outermost_points[:, 1],
                         color=to_hex(contours_color_area_palette[counter]),
                         linewidth=10, zorder=1
                         )
                plt.plot(outermost_points[:, 0],
                         outermost_points[:, 1],
                         color=to_hex(contours_color_palette[counter]),
                         label=str(cluster_name), linewidth=2, zorder=2
                         )
                counter += 1

    # Scatter plot 3
    colors = ['#7f83fc' if value == 0 else '#ff9f9f' for value in test_misclassifications]
    plt.scatter(
        x=misclassifier_labeled_test_umap_features[:, 0],
        y=misclassifier_labeled_test_umap_features[:, 1],
        c=colors,
        marker="X",
        zorder=3,
        linewidth=0.5,
        edgecolor="black",
        alpha=1
    )

    ax.axis()

    # Update legend
    handles, labels = ax.get_legend_handles_labels()

    # Apply the initial mapping for the first three indices and the rest with cluster
    new_labels = [
        INITIAL_LABEL_MAPPING.get(normalize_label(label), f'Cluster {label}')
        if idx < 3 else label_mapping.get(label, f'Cluster {label}')
        for idx, label in enumerate(labels)
    ]

    non_ctc_patch = mlines.Line2D([], [], color='#7f83fc', marker='X', linestyle='None',
                                  markersize=10, markeredgecolor='black', markeredgewidth=0.5,
                                  label='Non-CTC prediction (GT: CTC)')
    ctc_patch = mlines.Line2D([], [], color='#ff9f9f', marker='X', linestyle='None',
                              markersize=10, markeredgecolor='black', markeredgewidth=0.5,
                              label='CTC prediction (GT:Non-CTC)')

    handles.extend([non_ctc_patch, ctc_patch])
    new_labels.extend(['Non-CTC prediction (GT: CTC)', 'CTC prediction (GT:Non-CTC)'])

    legend = ax.legend(handles, new_labels, loc='upper left', title='Legend')
    legend.get_frame().set_facecolor('none')

    # Set dynamic limits based on data
    x_data = np.concatenate(
        [unlabeled_train_umap_features[:, 0], correct_classified_test_umap_features[:, 0],
         labeled_test_and_unlabeled_train_umap_features[:, 0], misclassifier_labeled_test_umap_features[:, 0]])
    y_data = np.concatenate(
        [unlabeled_train_umap_features[:, 1], correct_classified_test_umap_features[:, 1],
         labeled_test_and_unlabeled_train_umap_features[:, 1], misclassifier_labeled_test_umap_features[:, 1]])

    ax.set_xlim([x_data.min() - 1, x_data.max() + 1])
    ax.set_ylim([y_data.min() - 1, y_data.max() + 1])

    ax.set_xlabel(r'UMAP dimension 1', fontsize=15)
    ax.set_ylabel(r'UMAP dimension 2', fontsize=15)
    ax.set_title(title, fontsize=20, y=1.05)

    # Make the entire figure background transparent
    fig.patch.set_alpha(0)
    ax.patch.set_alpha(0)  # Ensure the axis patch is transparent

    # Set axis frame (rectangle) visible
    for spine in ax.spines.values():
        spine.set_edgecolor('black')  # Set the frame border color
        spine.set_linewidth(0.8)

    if run is not None and loop is not None:
        plt.savefig(
            os.path.join(save_path, "plots", f"hdbscan_with_contours_run_{run}_loop_{loop}.pdf"),
            dpi=700,
            facecolor='none',
            edgecolor='black'
        )
    else:
        plt.savefig(
            os.path.join(save_path, "plots", f"hdbscan_with_contours.pdf"),
            dpi=700,
            facecolor='none',
            edgecolor='black'
        )

    plt.close()
