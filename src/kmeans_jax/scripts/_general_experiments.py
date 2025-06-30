import logging
import os
from typing import Dict
from typing_extensions import Literal

import jax
import jax.numpy as jnp
import matplotlib as mpl
import numpy as np
from jaxtyping import Array, Bool, Float, Int, PRNGKeyArray
from sklearn import metrics as sk_metrics
from tqdm import tqdm

from ..hartigan_kmeans import (
    run_batched_hartigan_kmeans,
    run_hartigan_kmeans,
)
from ..kmeans import (
    assign_clusters,
    compute_loss,
    kmeans_plusplus_init,
    kmeans_random_init,
    run_kmeans,
    update_centroids,
)
from ..svd_utils import principal_component_analysis


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
    n_clusters: Int,
    size_clusters: Int[Array, " n_clusters"],
    dimension: Int,
    num_pca_components: Int,
    init_method: Literal["random_centers", "kmeans++", "random_partition"],
    var_prior: Float,
    max_iters: Int,
) -> Dict[str, Float]:
    key_centers, key_noise, key_pca, key_init = jax.random.split(key, 4)

    # Generate data
    true_centers = jax.random.normal(
        key_centers, shape=(n_clusters, dimension)
    ) * jnp.sqrt(var_prior)
    true_labels = jnp.arange(n_clusters).repeat(size_clusters)

    data = true_centers[true_labels] + jax.random.normal(
        key_noise, shape=(true_labels.shape[0], dimension)
    ) * jnp.sqrt(noise_variance)

    true_data_averages = update_centroids(data, true_labels, n_clusters)
    true_loss = compute_loss(data, true_data_averages, true_labels)

    # PCA
    data_pca = principal_component_analysis(
        key=key_pca, data=data, n_components=num_pca_components, mode="randomized"
    )

    ### Initialization
    if init_method == "random_centers":
        init_centroids, indices = kmeans_random_init(data, n_clusters, key_init)
        init_centroids_pca = data_pca[indices]
        init_partition = assign_clusters(init_centroids, data)

    elif init_method == "kmeans++":
        init_centroids, indices = kmeans_plusplus_init(data, n_clusters, key_init)
        init_centroids_pca = data_pca[indices]
        init_partition = assign_clusters(init_centroids, data)

    elif init_method == "random_partition":
        init_partition = jax.random.choice(
            key_init, true_labels, shape=(data.shape[0],), replace=False
        )
        init_centroids = update_centroids(data, init_partition, n_clusters)
        init_centroids_pca = update_centroids(data_pca, init_partition, n_clusters)

    else:
        raise ValueError(
            f"Unknown initialization method: {init_method}"
            + "Only 'random_centers', 'kmeans++', and 'random_partition' are supported."
        )

    # Regular k-means
    (_, labels), losses = run_kmeans(data, init_centroids, max_iters=max_iters)
    nmi_kmeans = sk_metrics.normalized_mutual_info_score(true_labels, labels)

    # Hartigan k-means
    (_, labels_hartigan), losses_hartigan = run_hartigan_kmeans(
        data, init_centroids, max_iters=max_iters
    )
    nmi_hartigan = sk_metrics.normalized_mutual_info_score(true_labels, labels_hartigan)

    # Batched Hartigan k-means
    (_, labels_bhartigan), losses_bhartigan = run_batched_hartigan_kmeans(
        data, init_centroids, max_iters=max_iters
    )
    nmi_bhartigan = sk_metrics.normalized_mutual_info_score(true_labels, labels_bhartigan)

    # print(losses.shape)
    # PCA + k-means
    (_, labels_pca), _ = run_kmeans(data_pca, init_centroids_pca, max_iters=max_iters)
    nmi_kmeans_pca = sk_metrics.normalized_mutual_info_score(true_labels, labels_pca)
    loss_pca = compute_loss(
        data, update_centroids(data, labels_pca, n_clusters), labels_pca
    )

    results = {
        "nmi": (nmi_kmeans, nmi_hartigan, nmi_bhartigan, nmi_kmeans_pca),
        "loss": (
            losses[-1],
            losses_hartigan[-1],
            losses_bhartigan[-1],
            loss_pca,
            true_loss,
        ),
    }
    return results


def run_general_experiments(
    dimension_vals: Int[Array, " n_dims"],
    noise_variance_vals: Float[Array, " n_noise_variances"],
    prior_variance: Float,
    n_clusters: Int,
    size_clusters: Int[Array, " n_clusters"],
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
    assert n_clusters > 1
    assert jnp.all(size_clusters > 0)
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
        "nmi_hartigan": np.zeros(shape_outputs),
        "nmi_bhartigan": np.zeros(shape_outputs),
        "nmi_kmeans_pca": np.zeros(shape_outputs),
        "loss_kmeans": np.zeros(shape_outputs),
        "loss_hartigan": np.zeros(shape_outputs),
        "loss_bhartigan": np.zeros(shape_outputs),
        "loss_kmeans_pca": np.zeros(shape_outputs),
        "loss_true_partition": np.zeros(shape_outputs),
        # experiment parameters
        "dimension_vals": dimension_vals,
        "noise_variance_vals": noise_variance_vals,
        "prior_variance": prior_variance,
        "n_clusters": n_clusters,
        "size_clusters": size_clusters,
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
                    n_clusters=n_clusters,
                    size_clusters=size_clusters,
                    dimension=dimension_vals[i],
                    num_pca_components=num_pca_components,
                    init_method=init_method,
                    var_prior=prior_variance,
                    max_iters=max_iters,
                )

                results["nmi_kmeans"][i, j, k] = experiment_result["nmi"][0]
                results["nmi_hartigan"][i, j, k] = experiment_result["nmi"][1]
                results["nmi_bhartigan"][i, j, k] = experiment_result["nmi"][2]
                results["nmi_kmeans_pca"][i, j, k] = experiment_result["nmi"][3]
                results["loss_kmeans"][i, j, k] = experiment_result["loss"][0]
                results["loss_hartigan"][i, j, k] = experiment_result["loss"][1]
                results["loss_bhartigan"][i, j, k] = experiment_result["loss"][2]
                results["loss_kmeans_pca"][i, j, k] = experiment_result["loss"][3]
                results["loss_true_partition"][i, j, k] = experiment_result["loss"][4]

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
