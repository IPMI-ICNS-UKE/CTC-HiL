import os
import random

from typing import List, Dict, Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import to_rgba

initial_label_mapping = {
    '10.0': 'Unlabeled train data',
    '0.0': 'Non-CTC (test data)',
    '1.0': 'CTC (test data)',
    '-1': 'Background'
}

predefined_colors_palette = ['#edbdb0', '#dfc181', '#ce92c8', 'g', '#95b282', '#828090']
predefined_area_palette = ['#f3d5cd', '#f6eedd', '#f2d1ef', '#e3f3e5', '#c2d0b6', '#cfcce5']


def generate_random_color(random_gen: random.Random) -> List[float]:
    """
    Generate a random color represented as a list of RGB float values.

    Params:
        random_gen (random.Random): A random generator instance to ensure reproducibility.

    Returns:
        List[float]: A list with three elements representing RGB values in the range [0.0, 1.0].
    """
    color = [random_gen.random() for _ in range(3)]
    return color


def lighten_color(color, amount=0.5) -> np.ndarray:
    """
    Lighten a given color by blending it with white.

    Parameters:
        color (Union[str, np.ndarray]): The input color represented as an RGBA string or array.
        amount (float): The weight factor for blending. Default is 0.5 for equal blend.

    Returns:
        np.ndarray: A lighter version of the input color as an RGBA NumPy array.
    """
    c = np.array(to_rgba(color))
    white = np.array([1, 1, 1, 0])
    return (1 - amount) * c + amount * white


def get_cluster_color(cluster_name: int, random_gen: random.Random):
    """
    Retrieve the color for the given cluster name. If the cluster does not have
    an assigned color, a new random color is generated and assigned.

    Params:
        cluster_name (int): The identifier for the cluster.
        random_gen (random.Random): A random generator instance for color generation.

    Returns:
        str: A string representing the color assigned to the cluster.
    """
    cluster_color_mapping = {}
    if cluster_name in cluster_color_mapping:
        return cluster_color_mapping[cluster_name]
    else:
        random_color = generate_random_color(random_gen)
        cluster_color_mapping[cluster_name] = random_color
        return random_color


def prepare_contour_colors_and_labels(
        cluster_names: List,
        predefined_label_mapping: Dict,
        predefined_colors: List,
        predefined_area_colors: List,
        random_state=42
) -> Tuple[List, List, Dict]:
    """
    Prepare color palettes and labels for cluster contours.

    Parameters:
        cluster_names (List): List of cluster identifiers/ names.
        predefined_label_mapping (Dict): Existing mapping of cluster labels.
        predefined_colors (List): List of predefined colors for contours.
        predefined_area_colors (List]): List of predefined area colors for contours.
        random_state: Seed for random number generator to ensure reproducibility. Default is 42.

    Returns:
        Tuple[List, List, Dict]:
            - First list contains colors for cluster contours.
            - Second list contains area colors for clusters.
            - Dictionary with updated mappings of cluster names to labels.
    """
    label_mapping = predefined_label_mapping.copy()
    contours_color_palette = predefined_colors.copy()
    contours_color_area_palette = predefined_area_colors.copy()

    random_gen = random.Random(random_state)  # Set a fixed seed for reproducibility

    for cluster_name in cluster_names:
        str_cluster_name = str(cluster_name)
        if str_cluster_name not in label_mapping:
            random_color = get_cluster_color(cluster_name, random_gen)
            lightened_color = lighten_color(random_color)
            contours_color_palette.append(random_color)
            contours_color_area_palette.append(lightened_color)
            label_mapping[str_cluster_name] = f'Cluster {str_cluster_name}'

    return contours_color_palette[:len(cluster_names)], contours_color_area_palette[:len(cluster_names)], label_mapping


def plot_line_plot_s1(
        df_results: pd.DataFrame,
        title: str,
        file_name: str,
        save_path: str,
        fig_size=(15, 15)
) -> None:
    """
    Generate and save a line plot visualizing F1 scores for different clusters over loops.

    Params:
        df_results (pd.DataFrame): DataFrame containing F1 scores and clusters data.
        title (str): Title for the plot.
        file_name (str): Name of the file to save the plot.
        save_path (str): Directory path to save the plot.
        fig_size (Tuple[int, int]): Size of the figure for the plot.
    """
    # Setting figure size
    sns.set(rc={'figure.figsize': fig_size})

    unique_clusters = np.unique(df_results['cluster name'])
    cluster_order = sorted(unique_clusters)
    complete_markers = ['o'] * len(cluster_order)

    # # Handling color palettes based on cluster counts
    if len(unique_clusters) <= len(predefined_colors_palette):
        # Slice the palettes and markers based on the number of unique clusters
        if -1 not in cluster_order:
            cluster_order = list(np.concatenate(([-1], cluster_order)))
        mean_line_palette = [predefined_colors_palette[cluster_order.index(cluster)] for cluster in unique_clusters]
        std_area_palette = [predefined_area_palette[cluster_order.index(cluster)] for cluster in unique_clusters]
        label_mapping = initial_label_mapping.copy()
        for cluster_name in cluster_order:
            str_cluster_name = str(cluster_name)
            if str_cluster_name not in label_mapping:
                label_mapping[str_cluster_name] = f'Cluster {str_cluster_name}'
    # If the list is longer than the predefined color list, then new colors are added
    else:
        mean_line_palette, std_area_palette, label_mapping = prepare_contour_colors_and_labels(
            cluster_order, initial_label_mapping, predefined_colors_palette, predefined_area_palette)

    df_results["loop"] = df_results["loop"].astype(str)

    # Plot lines
    line_plot = sns.lineplot(
        x='loop',
        y="f1 mean",
        data=df_results,
        hue="cluster name",
        palette=mean_line_palette,
        linewidth=2.5,
        style="cluster name",
        markers=complete_markers,
        markersize=7,
        dashes=False
    )

    # Fill areas for standard deviation
    for cluster_name, color in zip(unique_clusters, std_area_palette):
        cluster_data = df_results[df_results['cluster name'] == cluster_name]
        plt.fill_between(cluster_data['loop'], cluster_data["f1 mean"] - cluster_data["f1 std"],
                         cluster_data["f1 mean"] + cluster_data["f1 std"], color=color, alpha=0.7)

    for spine in plt.gca().spines.values():
        spine.set_edgecolor('black')

    # Annotate plot
    for line in range(0, df_results.shape[0]):
        mean_val = df_results["f1 mean"][line]
        mean_val_formatted = "0.00" if mean_val == 0.0 else "{:.2f}".format(mean_val)
        try:
            x_pos = float(df_results['loop'][line]) - 0.07
        except ValueError:
            x_pos = -0.07
        line_plot.text(x_pos, mean_val + 0.007, mean_val_formatted, horizontalalignment='left',
                       size='medium', color='black', weight='semibold')

    # Set plot labels and titles
    line_plot.set_title(title, pad=20, fontsize=20)
    line_plot.yaxis.labelpad = 20
    line_plot.xaxis.labelpad = 20
    line_plot.set_xlabel("Loop", fontsize=20)
    line_plot.tick_params(axis='x', labelsize=20)
    line_plot.set_ylabel("F1 score", fontsize=20)
    line_plot.tick_params(axis='y', labelsize=20)

    # Adjust x-ticks
    unique_loops = df_results['loop'].unique()
    x_ticks = [-0.07 if val == 'init' else float(val) for val in unique_loops]
    x_labels = ['init' if val == 'init' else str(val) for val in unique_loops]
    plt.xticks(ticks=x_ticks, labels=x_labels)
    plt.yticks(np.arange(0, 1.1, 0.1))

    # Additional grid and background settings
    plt.grid(axis='y', color='gray', linestyle='-', linewidth=0.5)
    line_plot.set_facecolor('white')

    # Update legend
    handles, labels = line_plot.get_legend_handles_labels()
    new_labels = [label_mapping.get(label, label) for label in labels]
    line_plot.legend(handles, new_labels, loc='upper left')
    sns.move_legend(line_plot, "upper left", bbox_to_anchor=(1, 1))

    # Save plot to file
    output_file = os.path.join(save_path, f"{file_name}.pdf")
    plt.savefig(output_file, dpi=1200)
    plt.close()
