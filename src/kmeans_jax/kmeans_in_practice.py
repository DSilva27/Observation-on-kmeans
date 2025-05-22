import logging
import os
from typing import Dict
from typing_extensions import Literal

import jax
import jax.numpy as jnp
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from jaxtyping import Array, Bool, Float, Int, PRNGKeyArray
from sklearn import metrics as sk_metrics
from sklearn.decomposition import PCA
from tqdm import tqdm

from .kmeans import (
    assign_clusters,
    compute_loss,
    kmeans_plusplus_init,
    kmeans_random_init,
    run_kmeans,
    update_centroids,
)


mpl.rcParams["pdf.fonttype"] = 42  # TrueType fonts
mpl.rcParams["ps.fonttype"] = 42


def _mkbasedir(path: str) -> None:
    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except (FileExistsError, PermissionError):
            raise ValueError("Output path does not exist or cannot be created.")
    return


def run_single_experiment(
    key: PRNGKeyArray,
    noise_variance: Float,
    size_cluster1: Int,
    size_cluster2: Int,
    dimension: Int,
    num_pca_components: Int,
    init_method: Literal["random_centers", "kmeans++", "random_partition"],
    var_prior: Float,
    max_iters: Int,
) -> Dict[str, Float]:
    key1, key2, key_data1, key_data2, key_init = jax.random.split(key, 5)

    # Generate data
    true_mu_C = jax.random.normal(key1, shape=(dimension,)) * jnp.sqrt(var_prior)
    true_mu_T = jax.random.normal(key2, shape=(dimension,)) * jnp.sqrt(var_prior)

    x_C = true_mu_C[None, ...] + jax.random.normal(
        key_data1, shape=(size_cluster1, dimension)
    ) * jnp.sqrt(noise_variance)

    x_T = true_mu_T[None, ...] + jax.random.normal(
        key_data2, shape=(size_cluster2, dimension)
    ) * jnp.sqrt(noise_variance)

    data = jnp.concatenate([x_C, x_T])
    true_labels = jnp.concatenate(
        [jnp.zeros(size_cluster1, dtype=int), jnp.ones(size_cluster2, dtype=int)]
    )
    true_data_averages = update_centroids(data, true_labels, 2)
    true_loss = compute_loss(data, true_data_averages, true_labels)

    # PCA
    pca = PCA(
        n_components=num_pca_components,
    )
    pca.fit(np.array(data))
    data_pca = jnp.array(pca.transform(np.array(data)))

    ### Initialization

    if init_method == "random_centers":
        init_centroids, indices = kmeans_random_init(data, 2, key_init)
        init_centroids_pca = data_pca[indices]
        init_partition = assign_clusters(init_centroids, data)

    elif init_method == "kmeans++":
        init_centroids, indices = kmeans_plusplus_init(data, 2, key_init)
        init_centroids_pca = data_pca[indices]
        init_partition = assign_clusters(init_centroids, data)

    elif init_method == "random_partition":
        init_partition = jax.random.choice(
            key_init, true_labels, shape=(data.shape[0],), replace=False
        )
        init_centroids = update_centroids(data, init_partition, 2)
        init_centroids_pca = update_centroids(data_pca, init_partition, 2)

    else:
        raise ValueError(
            f"Unknown initialization method: {init_method}"
            + "Only 'random_centers', 'kmeans++', and 'random_partition' are supported."
        )

    # Regular k-means
    (_, labels), losses = run_kmeans(data, init_centroids, max_iters=max_iters)
    nmi_kmeans = sk_metrics.normalized_mutual_info_score(true_labels, labels)

    # print(losses.shape)
    # PCA + k-means
    (_, labels_pca), _ = run_kmeans(data_pca, init_centroids_pca, max_iters=max_iters)
    nmi_kmeans_pca = sk_metrics.normalized_mutual_info_score(true_labels, labels_pca)
    loss_pca = compute_loss(data, update_centroids(data, labels_pca, 2), labels_pca)

    # Split PCA
    labels_pca_split = jnp.where(data_pca[:, 0] > 0.0, 0, 1).astype(int)
    nmi_split_pca = sk_metrics.normalized_mutual_info_score(true_labels, labels_pca_split)
    loss_pca_split = compute_loss(
        data, update_centroids(data, labels_pca_split, 2), labels_pca_split
    )
    results = {
        "nmi": (nmi_kmeans, nmi_kmeans_pca, nmi_split_pca),
        "loss": (losses[-1], loss_pca, loss_pca_split, true_loss),
    }
    return results


def run_kmeans_in_practice_experiments(
    dimension_vals: Int[Array, " n_dims"],
    noise_variance_vals: Float[Array, " n_noise_variances"],
    prior_variance: Float,
    size_cluster1: Int,
    size_cluster2: Int,
    n_experiments: Int,
    num_pca_components: Int,
    init_method: Literal["random_centers", "kmeans++", "random_partition"],
    path_to_output: str,
    *,
    max_iters: Int = 1000,
    seed: Int = 0,
    overwrite: Bool = False,
) -> Dict[str, Float[Array, "n_dims n_noise_variances n_experiments"]]:
    """
    Run k-means in practice experiments.
    **Arguments:**
        - dimension_vals: Array of dimensions to test.
        - noise_variance_vals: Array of noise variances to test.
        - prior_variance: Prior variance for the data generation.
        - size_cluster1: Size of the first cluster.
        - size_cluster2: Size of the second cluster.
        - n_experiments: Number of experiments to run for each setting.
        - num_pca_components: Number of PCA components to use.
        - init_method: Initialization method for k-means.
            One of 'random_centers', 'kmeans++', or 'random_partition'.
        - path_to_output: Path to save the results.
        - max_iters: Maximum number of iterations for k-means.
        - seed: Random seed for reproducibility.
        - overwrite: Whether to overwrite existing output files.
    **Returns:**
        - results: Dictionary containing the results of the experiments.
        The results are the NMI vs the true labels, and loss values for each experiment.
    """
    assert jnp.all(dimension_vals > 0)
    assert jnp.all(noise_variance_vals > 0)
    assert jnp.all(prior_variance > 0)
    assert jnp.all(size_cluster1 > 0)
    assert jnp.all(size_cluster2 > 0)
    assert jnp.all(n_experiments > 0)
    assert jnp.all(num_pca_components > 0)
    assert jnp.all(max_iters > 0)

    assert init_method in [
        "random_centers",
        "kmeans++",
        "random_partition",
    ], (
        "Unknown initialization method. "
        + "Only 'random_centers', 'kmeans++', and 'random_partition' are supported."
    )
    if os.path.exists(path_to_output) and not overwrite:
        raise ValueError(
            f"Ouput file {path_to_output} exists, but overwrite was set to False"
        )

    else:
        basedir = os.path.dirname(path_to_output)
        _mkbasedir(basedir)

    key = jax.random.key(seed)

    logging.info("Starting experiments")
    logging.info("=" * 100)

    shape_outputs = (
        len(dimension_vals),
        len(noise_variance_vals),
        n_experiments,
    )
    results = {
        # experiment results
        "nmi_kmeans": np.zeros(shape_outputs),
        "nmi_kmeans_pca": np.zeros(shape_outputs),
        "nmi_split_pca": np.zeros(shape_outputs),
        "loss_kmeans": np.zeros(shape_outputs),
        "loss_kmeans_pca": np.zeros(shape_outputs),
        "loss_split_pca": np.zeros(shape_outputs),
        "loss_true_partition": np.zeros(shape_outputs),
        # experiment parameters
        "dimension_vals": dimension_vals,
        "noise_variance_vals": noise_variance_vals,
        "prior_variance": prior_variance,
        "size_cluster1": size_cluster1,
        "size_cluster2": size_cluster2,
        "n_experiments": n_experiments,
        "num_pca_components": num_pca_components,
        "init_method": init_method,
        "max_iters": max_iters,
        # run information
        "i": 0,
        "j": 0,
    }

    for i in tqdm(range(len(dimension_vals))):
        results["i"] = i
        logging.info(f"  Running for d = {dimension_vals[i]}")
        for j in range(len(noise_variance_vals)):
            results["j"] = j
            logging.info(
                f"    Running for noise_variance_vals = {noise_variance_vals[j]}"
            )

            logging.info("      Running experiments")
            for k in range(n_experiments):
                key, subkey = jax.random.split(key)
                experiment_result = run_single_experiment(
                    key=subkey,
                    noise_variance=noise_variance_vals[j],
                    size_cluster1=size_cluster1,
                    size_cluster2=size_cluster2,
                    dimension=dimension_vals[i],
                    num_pca_components=num_pca_components,
                    init_method=init_method,
                    var_prior=prior_variance,
                    max_iters=max_iters,
                )

                results["nmi_kmeans"][i, j, k] = experiment_result["nmi"][0]
                results["nmi_kmeans_pca"][i, j, k] = experiment_result["nmi"][1]
                results["nmi_split_pca"][i, j, k] = experiment_result["nmi"][2]
                results["loss_kmeans"][i, j, k] = experiment_result["loss"][0]
                results["loss_kmeans_pca"][i, j, k] = experiment_result["loss"][1]
                results["loss_split_pca"][i, j, k] = experiment_result["loss"][2]
                results["loss_true_partition"][i, j, k] = experiment_result["loss"][3]

            logging.info("      Done running experiments. Moving to next setting.")
            logging.info("=" * 100)
            jax.clear_caches()

            jnp.savez(
                path_to_output,
                **results,
            )

            logging.info(f"Saved preliminary results to {path_to_output}")
    logging.info("Finished running all experiments.")
    return results


def plot_kmeans_in_practice_nmi_results(results, fig_fname=None, fig_suptitle=None):
    # sns.set_theme(context="talk")

    def _plot_nmi(nmi_matrix, ax, method):
        im = ax.imshow(
            nmi_matrix.mean(-1).T, origin="lower", vmin=0, vmax=1, cmap="cividis"
        )
        ax.set_xticks(tick_indices, tick_labels, fontsize=fs)

        ax.set_yticks(
            np.arange(0, noise_variance_vals.shape[0], 2),
            labels=noise_variance_vals[::2].round(decimals=1),
            fontsize=fs,
        )
        ax.set_xlabel("d [dimension]", fontsize=fs)
        ax.set_title(method, fontsize=fs)

        return im

    fs = 16

    noise_variance_vals = results["noise_variance_vals"]
    dimension_vals = results["dimension_vals"]

    # x-axis ticks
    tick_indices = np.linspace(0, len(dimension_vals) - 1, num=5, dtype=int)
    tick_labels = [f"{d:.0f}" for d in np.log10(dimension_vals[tick_indices])]
    tick_labels = [rf"$10^{d}$" for d in tick_labels]

    fig, ax = plt.subplots(1, 3, figsize=(15, 4), sharey=True, layout="compressed")
    im = _plot_nmi(results["nmi_kmeans"], ax[0], "k-means")
    im = _plot_nmi(results["nmi_kmeans_pca"], ax[1], "PCA + k-means")
    im = _plot_nmi(results["nmi_split_pca"], ax[2], "PCA + Split")

    ax[0].set_ylabel(r"$\sigma^2$ [noise variance]", fontsize=fs)
    cbar = plt.colorbar(im)
    cbar.ax.tick_params(labelsize=12)
    cbar.set_label("Normalized Mutual Information", size=14)

    if fig_suptitle is not None:
        fig.suptitle(fig_suptitle, fontsize=fs)

    if fig_fname is not None:
        plt.savefig(fig_fname, bbox_inches="tight", dpi=300)

    return fig, ax


def plot_kmeans_in_practice_loss_results(results, fig_fname=None, fig_suptitle=None):
    def _compute_loss_metric(loss1, loss2):
        return jnp.where(
            jnp.isclose(loss1, loss2), 0, jnp.where(loss1 - loss2 < 0, 1, -1)
        ).mean(-1)

    def _plot_loss(loss_matrix, ax, method):
        im = ax.imshow(loss_matrix.T, origin="lower", vmin=-1, vmax=1, cmap="bwr_r")
        ax.set_xticks(tick_indices, tick_labels, fontsize=fs)

        ax.set_yticks(
            np.arange(0, noise_variance_vals.shape[0], 2),
            labels=noise_variance_vals[::2].round(decimals=1),
            fontsize=fs,
        )
        ax.set_xlabel("d [dimension]", fontsize=fs)
        ax.set_title(method, fontsize=fs)

        return im

    fs = 16

    noise_variance_vals = results["noise_variance_vals"]
    dimension_vals = results["dimension_vals"]

    # x-axis ticks
    tick_indices = np.linspace(0, len(dimension_vals) - 1, num=5, dtype=int)
    tick_labels = [f"{d:.0f}" for d in np.log10(dimension_vals[tick_indices])]
    tick_labels = [rf"$10^{d}$" for d in tick_labels]

    fig, ax = plt.subplots(1, 3, figsize=(15, 4), sharey=True, layout="compressed")
    im = _plot_loss(
        _compute_loss_metric(results["loss_kmeans"], results["loss_true_partition"]),
        ax[0],
        "k-means",
    )
    im = _plot_loss(
        _compute_loss_metric(results["loss_kmeans_pca"], results["loss_true_partition"]),
        ax[1],
        "PCA + k-means",
    )
    im = _plot_loss(
        _compute_loss_metric(results["loss_split_pca"], results["loss_true_partition"]),
        ax[2],
        "PCA + Split",
    )

    ax[0].set_ylabel(r"$\sigma^2$ [noise variance]", fontsize=fs)
    cbar = plt.colorbar(im)
    cbar.ax.set_yticks([-1.0, 0.0, 1.0])  # Set ticks explicitly
    cbar.ax.set_yticklabels(["Ground Truth", "Tie", "Clustering"])  # Set matching labels
    cbar.set_label("Partition with better loss", size=14)

    if fig_suptitle is not None:
        fig.suptitle(fig_suptitle, fontsize=fs)

    if fig_fname is not None:
        plt.savefig(fig_fname, bbox_inches="tight", dpi=300)

    return fig, ax
