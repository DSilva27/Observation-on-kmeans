import argparse
import logging
import os
from typing import Dict, Tuple
from typing_extensions import Literal

import jax
import jax.numpy as jnp
import numpy as np
import yaml
from jaxtyping import Array, Bool, Float, Int, PRNGKeyArray
from sklearn import metrics as sk_metrics
from tqdm import tqdm

from ..kmeans import KMeans
from ..kmeans._common_functions import (
    compute_loss,
    compute_centroids,
)
from ..svd_utils import principal_component_analysis


DEFAULT_PARAMETERS = {
    "dimension_values": np.logspace(0.8, 7, 15, dtype=int)[:11],
    "prior_variance": 1.0,
    "noise_variance_vals": np.linspace(2.0, 6.0, 10) ** 2,
    "n_experiments": 100,
    "n_inits_per_experiment": 10,
    "num_pca_components": 2,
    "max_iter": 1000,
    "seed": 0,
}


def _mkbasedir(path: str) -> None:
    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except (FileExistsError, PermissionError):
            raise ValueError("Output path does not exist or cannot be created.")
    return


def _generate_data(
    key,
    noise_variance: Float,
    n_clusters: Int,
    size_clusters: Int[Array, " n_clusters"],
    dimension: Int,
    prior_variance: Float,
) -> Tuple[Float[Array, " n d"], Float[Array, " n_clusters d"], Int[Array, " n"]]:
    key_centers, key_noise = jax.random.split(key, 2)
    true_centers = jax.random.normal(
        key_centers, shape=(n_clusters, dimension)
    ) * jnp.sqrt(prior_variance)
    true_labels = jnp.arange(n_clusters).repeat(size_clusters)

    data = true_centers[true_labels] + jax.random.normal(
        key_noise, shape=(true_labels.shape[0], dimension)
    ) * jnp.sqrt(noise_variance)

    return data, true_labels


def run_single_experiment(
    key: PRNGKeyArray,
    noise_variance: Float,
    n_clusters: Int,
    size_clusters: Int[Array, " n_clusters"],
    dimension: Int,
    num_pca_components: Int,
    init_method: Literal["random", "kmeans++", "random partition"],
    n_init: Int,
    prior_variance: Float,
    max_iter: Int,
    *,
    batch_size: Int = None,
) -> Dict[str, Float]:
    key_data, key_pca, key_run = jax.random.split(key, 3)

    # Generate data
    data, true_labels = _generate_data(
        key=key_data,
        noise_variance=noise_variance,
        n_clusters=n_clusters,
        size_clusters=size_clusters,
        dimension=dimension,
        prior_variance=prior_variance,
    )

    true_data_averages = compute_centroids(data, true_labels, n_clusters)
    true_loss = compute_loss(data, true_data_averages, true_labels)

    # PCA
    data_pca = principal_component_analysis(
        key=key_pca, data=data, n_components=num_pca_components, mode="randomized"
    )

    lloyd_kmeans = KMeans(
        n_clusters=n_clusters,
        n_init=n_init,
        max_iter=max_iter,
        init=init_method,
        algorithm="Lloyd",
    )

    hartigan_kmeans = KMeans(
        n_clusters=n_clusters,
        n_init=n_init,
        max_iter=max_iter,
        init=init_method,
        algorithm="Hartigan",
    )

    bhartigan_kmeans = KMeans(
        n_clusters=n_clusters,
        n_init=n_init,
        max_iter=max_iter,
        init=init_method,
        algorithm="Batched Hartigan",
    )

    minibhartigan_kmeans = KMeans(
        n_clusters=n_clusters,
        n_init=n_init,
        max_iter=max_iter,
        init=init_method,
        algorithm="Mini-batch Hartigan",
        batch_size=int(size_clusters.sum() // 2),
    )

    # Run k-means
    lloyd_results = lloyd_kmeans.fit(key_run, data, output="best", batch_size=batch_size)
    hartigan_results = hartigan_kmeans.fit(
        key_run, data, output="best", batch_size=batch_size
    )
    bhartigan_results = bhartigan_kmeans.fit(
        key_run, data, output="best", batch_size=batch_size
    )
    lloyd_pca_results = lloyd_kmeans.fit(
        key_run, data_pca, output="best", batch_size=batch_size
    )
    minibhartigan_results = minibhartigan_kmeans.fit(
        key_run, data, output="best", batch_size=batch_size
    )

    # Compute the NMI
    nmi_kmeans = sk_metrics.normalized_mutual_info_score(
        true_labels, lloyd_results["labels"]
    )
    nmi_hartigan = sk_metrics.normalized_mutual_info_score(
        true_labels, hartigan_results["labels"]
    )
    nmi_kmeans_pca = sk_metrics.normalized_mutual_info_score(
        true_labels, lloyd_pca_results["labels"]
    )

    nmi_bhartigan = sk_metrics.normalized_mutual_info_score(
        true_labels, bhartigan_results["labels"]
    )

    nmi_minibhartigan = sk_metrics.normalized_mutual_info_score(
        true_labels, minibhartigan_results["labels"]
    )

    loss_pca = compute_loss(
        data,
        compute_centroids(data, lloyd_pca_results["labels"], n_clusters),
        lloyd_pca_results["labels"],
    )

    results = {
        "nmi": (
            nmi_kmeans,
            nmi_hartigan,
            nmi_bhartigan,
            nmi_minibhartigan,
            nmi_kmeans_pca,
        ),
        "loss": (
            lloyd_results["loss"],
            hartigan_results["loss"],
            bhartigan_results["loss"],
            minibhartigan_results["loss"],
            loss_pca,
            true_loss,
        ),
    }
    return results


def _run_general_experiments_from_scratch(
    dimension_vals: Int[Array, " n_dims"],
    noise_variance_vals: Float[Array, " n_noise_variances"],
    prior_variance: Float,
    n_clusters: Int,
    size_clusters: Int[Array, " n_clusters"],
    n_experiments: Int,
    n_inits_per_experiment: Int,
    num_pca_components: Int,
    init_method: Literal["random", "kmeans++", "random partition"],
    path_to_output: str,
    *,
    batch_size: Int = None,
    max_iter: Int = 1000,
    seed: Int = 0,
    overwrite: Bool = False,
) -> Dict[str, Float[Array, "n_dims n_noise_variances n_experiments"]]:
    """
    Run k-means in practice experiments.
    **Arguments:**
        - dimension_vals: Array of dimensions to test.
        - noise_variance_vals: Array of noise variances to test.
        - prior_variance: Prior variance for the data generation.
        - n_clusters: Number of clusters to use in the experiments.
        - size_clusters: Array of sizes of cluster sizes.
        - n_experiments: Number of experiments to run for each setting.
        - n_inits_per_experiment: Number of initializations per experiment.
        - num_pca_components: Number of PCA components to use.
        - init_method: Initialization method for k-means.
            One of 'random', 'kmeans++', or 'random partition'.
        - path_to_output: Path to save the results.
        - max_iter: Maximum number of iterations for k-means.
        - seed: Random seed for reproducibility.
        - overwrite: Whether to overwrite existing output fbests.
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
    assert jnp.all(max_iter > 0)

    assert init_method in [
        "random",
        "kmeans++",
        "random partition",
    ], (
        "Unknown initialization method. "
        + "Only 'random', 'kmeans++', and 'random partition' are supported."
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
        "nmi_minibhartigan": np.zeros(shape_outputs),
        "nmi_kmeans_pca": np.zeros(shape_outputs),
        "loss_kmeans": np.zeros(shape_outputs),
        "loss_hartigan": np.zeros(shape_outputs),
        "loss_bhartigan": np.zeros(shape_outputs),
        "loss_minibhartigan": np.zeros(shape_outputs),
        "loss_kmeans_pca": np.zeros(shape_outputs),
        "loss_true_partition": np.zeros(shape_outputs),
        # experiment parameters
        "dimension_vals": dimension_vals,
        "noise_variance_vals": noise_variance_vals,
        "prior_variance": prior_variance,
        "n_clusters": n_clusters,
        "size_clusters": size_clusters,
        "n_experiments": n_experiments,
        "n_inits_per_experiment": n_inits_per_experiment,
        "num_pca_components": num_pca_components,
        "init_method": init_method,
        "max_iter": max_iter,
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
                    n_init=n_inits_per_experiment,
                    prior_variance=prior_variance,
                    max_iter=max_iter,
                    batch_size=batch_size,
                )

                results["nmi_kmeans"][i, j, k] = experiment_result["nmi"][0]
                results["nmi_hartigan"][i, j, k] = experiment_result["nmi"][1]
                results["nmi_bhartigan"][i, j, k] = experiment_result["nmi"][2]
                results["nmi_minibhartigan"][i, j, k] = experiment_result["nmi"][3]
                results["nmi_kmeans_pca"][i, j, k] = experiment_result["nmi"][4]
                results["loss_kmeans"][i, j, k] = experiment_result["loss"][0]
                results["loss_hartigan"][i, j, k] = experiment_result["loss"][1]
                results["loss_bhartigan"][i, j, k] = experiment_result["loss"][2]
                results["loss_minibhartigan"][i, j, k] = experiment_result["loss"][3]
                results["loss_kmeans_pca"][i, j, k] = experiment_result["loss"][4]
                results["loss_true_partition"][i, j, k] = experiment_result["loss"][5]

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


def _run_general_experiments_continue(
    path_to_output: str,
    *,
    batch_size: Int = None,
    max_iter: Int = 1000,
    seed: Int = 0,
) -> Dict[str, Float[Array, "n_dims n_noise_variances n_experiments"]]:
    """
    Run k-means in practice experiments.
    **Arguments:**
        - dimension_vals: Array of dimensions to test.
        - noise_variance_vals: Array of noise variances to test.
        - prior_variance: Prior variance for the data generation.
        - n_clusters: Number of clusters to use in the experiments.
        - size_clusters: Array of sizes of cluster sizes.
        - n_experiments: Number of experiments to run for each setting.
        - n_inits_per_experiment: Number of initializations per experiment.
        - num_pca_components: Number of PCA components to use.
        - init_method: Initialization method for k-means.
            One of 'random', 'kmeans++', or 'random partition'.
        - path_to_output: Path to save the results.
        - max_iter: Maximum number of iterations for k-means.
        - seed: Random seed for reproducibility.
        - overwrite: Whether to overwrite existing output fbests.
    **Returns:**
        - results: Dictionary containing the results of the experiments.
        The results are the NMI vs the true labels, and loss values for each experiment.
    """
    if not os.path.exists(path_to_output):
        raise FileNotFoundError(
            f"Output file {path_to_output} does not exist. Cannot continue experiments."
        )

    else:
        results = dict(jnp.load(path_to_output, allow_pickle=True))

    dimension_vals = jnp.array(results["dimension_vals"])
    noise_variance_vals = jnp.array(results["noise_variance_vals"])
    prior_variance = float(results["prior_variance"])
    n_clusters = int(results["n_clusters"])
    size_clusters = jnp.array(results["size_clusters"], dtype=jnp.int32)
    n_experiments = results["n_experiments"]
    n_inits_per_experiment = int(results["n_inits_per_experiment"])
    num_pca_components = int(results["num_pca_components"])
    init_method = str(results["init_method"])

    curr_i = int(results["i"])
    curr_j = int(results["j"])

    key = jax.random.key(seed)

    logging.info(f"Continuing experiments from i = {curr_i}, j = {curr_j}")
    logging.info("=" * 100)

    for i in tqdm(range(curr_i, len(dimension_vals))):
        results["i"] = i
        logging.info(f"  Running for d = {dimension_vals[i]}")
        for j in range(curr_j, len(noise_variance_vals)):
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
                    n_init=n_inits_per_experiment,
                    prior_variance=prior_variance,
                    max_iter=max_iter,
                    batch_size=batch_size,
                )

                results["nmi_kmeans"][i, j, k] = experiment_result["nmi"][0]
                results["nmi_hartigan"][i, j, k] = experiment_result["nmi"][1]
                results["nmi_bhartigan"][i, j, k] = experiment_result["nmi"][2]
                results["nmi_minibhartigan"][i, j, k] = experiment_result["nmi"][3]
                results["nmi_kmeans_pca"][i, j, k] = experiment_result["nmi"][4]
                results["loss_kmeans"][i, j, k] = experiment_result["loss"][0]
                results["loss_hartigan"][i, j, k] = experiment_result["loss"][1]
                results["loss_bhartigan"][i, j, k] = experiment_result["loss"][2]
                results["loss_minibhartigan"][i, j, k] = experiment_result["loss"][3]
                results["loss_kmeans_pca"][i, j, k] = experiment_result["loss"][4]
                results["loss_true_partition"][i, j, k] = experiment_result["loss"][5]

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


def run_general_experiments(
    dimension_vals: Int[Array, " n_dims"],
    noise_variance_vals: Float[Array, " n_noise_variances"],
    prior_variance: Float,
    n_clusters: Int,
    size_clusters: Int[Array, " n_clusters"],
    n_experiments: Int,
    n_inits_per_experiment: Int,
    num_pca_components: Int,
    init_method: Literal["random", "kmeans++", "random partition"],
    path_to_output: str,
    *,
    batch_size: Int = None,
    max_iter: Int = 1000,
    seed: Int = 0,
    overwrite: Bool = False,
    continue_experiments: Bool = False,
) -> Dict[str, Float[Array, "n_dims n_noise_variances n_experiments"]]:
    """
    Run k-means in practice experiments.
    **Arguments:**
        - dimension_vals: Array of dimensions to test.
        - noise_variance_vals: Array of noise variances to test.
        - prior_variance: Prior variance for the data generation.
        - n_clusters: Number of clusters to use in the experiments.
        - size_clusters: Array of sizes of cluster sizes.
        - n_experiments: Number of experiments to run for each setting.
        - n_inits_per_experiment: Number of initializations per experiment.
        - num_pca_components: Number of PCA components to use.
        - init_method: Initialization method for k-means.
            One of 'random', 'kmeans++', or 'random partition'.
        - path_to_output: Path to save the results.
        - max_iter: Maximum number of iterations for k-means.
        - seed: Random seed for reproducibility.
        - overwrite: Whether to overwrite existing output files.
        - continue_experiments: Whether to continue from existing
            results or start from scratch.
    **Returns:**
        - results: Dictionary containing the results of the experiments.
        The results are the NMI vs the true labels, and loss values for each experiment.
    """
    if continue_experiments:
        return _run_general_experiments_continue(
            path_to_output=path_to_output,
            batch_size=batch_size,
            max_iter=max_iter,
            seed=seed,
        )
    else:
        return _run_general_experiments_from_scratch(
            dimension_vals=dimension_vals,
            noise_variance_vals=noise_variance_vals,
            prior_variance=prior_variance,
            n_clusters=n_clusters,
            size_clusters=size_clusters,
            n_experiments=n_experiments,
            n_inits_per_experiment=n_inits_per_experiment,
            num_pca_components=num_pca_components,
            init_method=init_method,
            path_to_output=path_to_output,
            batch_size=batch_size,
            max_iter=max_iter,
            seed=seed,
            overwrite=overwrite,
        )


def main(
    config,
    path_to_output: str,
    overwrite: bool = False,
    batch_size: int = None,
    continue_from_previous: bool = False,
) -> int:
    run_general_experiments(
        dimension_vals=config["dimension_values"],
        noise_variance_vals=config["noise_variance_vals"],
        prior_variance=config["prior_variance"],
        n_clusters=config["n_clusters"],
        size_clusters=config["size_clusters"],
        n_experiments=config["n_experiments"],
        n_inits_per_experiment=config["n_inits_per_experiment"],
        num_pca_components=config["num_pca_components"],
        init_method="kmeans++",
        max_iter=config["max_iter"],
        seed=config["seed"],
        path_to_output=path_to_output,
        overwrite=overwrite,
        batch_size=batch_size,
        continue_experiments=continue_from_previous,
    )
    return 0


def _parse_config(config: dict) -> Dict:
    """
    Parse the configuration file and return the parameters.
    """
    for key in DEFAULT_PARAMETERS:
        if key not in config:
            config[key] = DEFAULT_PARAMETERS[key]

    return config


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run general k-means experiments.")
    parser.add_argument(
        "--output", type=str, required=True, help="Output path for results."
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite existing output file."
    )

    parser.add_argument(
        "--config_file", type=str, required=True, help="Path to the configuration file."
    )
    parser.add_argument(
        "--continue",
        dest="continue_from_previous",
        action="store_true",
        help="Continue from last run",
    )
    args = parser.parse_args()

    config = yaml.safe_load(open(args.config_file, "r"))
    config = _parse_config(config)

    path_to_output = args.output
    overwrite = args.overwrite
    main(config, path_to_output, overwrite, args.continue_from_previous)
