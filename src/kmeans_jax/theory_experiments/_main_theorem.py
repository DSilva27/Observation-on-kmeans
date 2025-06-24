import os
from typing import Dict

import jax
import numpy as np
from jaxtyping import Array, Bool, Float, Int, PRNGKeyArray
from tqdm import tqdm


jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from ..kmeans import (
    assign_clusters,
    kmeans_random_init,
    update_centroids,
)


def _compute_rho(
    prior_variance: Float,
    noise_variance: Float,
    size_cluster_C: Float,
    size_cluster_T: Float,
) -> Float:
    num = (
        4
        * noise_variance
        * (size_cluster_C - 1)
        * (size_cluster_C**2)
        * size_cluster_T
        * (size_cluster_T + 1)
        * (size_cluster_C * (noise_variance + 2 * prior_variance) - 2 * prior_variance)
    )
    den = (
        -size_cluster_C * (noise_variance + 4 * prior_variance) * size_cluster_T
        + size_cluster_C**2
        * (noise_variance + 2 * (noise_variance + prior_variance) * size_cluster_T)
        + 2 * prior_variance * size_cluster_T
    ) ** 2
    return num / den


def _compute_upper_bound_main_theorem(
    dimension: Int,
    prior_variance: Float,
    noise_variance: Float,
    size_cluster_C: Int,
    size_cluster_T: Int,
) -> Float:
    return _compute_rho(
        prior_variance, noise_variance, size_cluster_C, size_cluster_T
    ) ** (dimension / 4)


def _run_experiment_main_theorem_worst(
    key: PRNGKeyArray,
    size_cluster_C: Int,
    size_cluster_T: Int,
    dimension: Int,
    prior_std: Float,
    noise_std: Float,
) -> Int:
    key1, key2, key_data1, key_data2, key_x = jax.random.split(key, 5)
    true_mu_C = jax.random.normal(key1, shape=(dimension,)) * prior_std
    true_mu_T = jax.random.normal(key2, shape=(dimension,)) * prior_std

    x_C = (
        true_mu_C[None, ...]
        + jax.random.normal(key_data1, shape=(size_cluster_C - 1, dimension)) * noise_std
    )
    x_T = (
        true_mu_T[None, ...]
        + jax.random.normal(key_data2, shape=(size_cluster_T, dimension)) * noise_std
    )

    x = true_mu_T + jax.random.normal(key_x, shape=(dimension,)) * noise_std

    x_C = jnp.concatenate([x_C, x[None, ...]])

    mu_C = jnp.mean(x_C, axis=0)
    mu_T = jnp.mean(x_T, axis=0)

    point_swaps = np.sum((x - mu_T) ** 2) - jnp.sum((x - mu_C) ** 2)

    output = jax.lax.cond(
        point_swaps < 0.0,
        lambda _: 1,
        lambda _: 0,
        operand=None,
    )
    return output


def _run_experiment_main_theorem_random(
    key: PRNGKeyArray,
    size_cluster_C: Int,
    size_cluster_T: Int,
    dimension: Int,
    prior_variance: Float,
    noise_variance: Float,
) -> Int:
    key1, key2, key_data1, key_data2, key_x, key_init = jax.random.split(key, 6)
    true_mu_C = jax.random.normal(key1, shape=(dimension,)) * prior_variance
    true_mu_T = jax.random.normal(key2, shape=(dimension,)) * prior_variance

    x_C = (
        true_mu_C[None, ...]
        + jax.random.normal(key_data1, shape=(size_cluster_C - 1, dimension))
        * noise_variance
    )
    x_T = (
        true_mu_T[None, ...]
        + jax.random.normal(key_data2, shape=(size_cluster_T, dimension)) * noise_variance
    )

    data = jnp.concatenate([x_C, x_T])
    init_centroids, _ = kmeans_random_init(data, 2, key_init)
    assignments1 = assign_clusters(init_centroids, data)
    centroids = update_centroids(data, assignments1, 2)

    idx = jax.random.choice(key_x, data.shape[0])
    x = data[idx]
    x_assign = assignments1[idx]
    x_not_assign = 1 - x_assign

    point_swaps = jnp.sum((x - centroids[x_not_assign]) ** 2) - jnp.sum(
        (x - centroids[x_assign]) ** 2
    )

    output = jax.lax.cond(
        point_swaps < 0.0,
        lambda _: 1,
        lambda _: 0,
        operand=None,
    )

    return output


def run_main_theorem_experiments(
    dimension_vals: Int[Array, " n_dim_vals"],
    noise_std_vals: Float[Array, " n_noise_std_vals"],
    prior_std: Float,
    size_cluster_C: Int,
    size_cluster_T: Int,
    n_experiments: Int,
    path_to_output: str,
    *,
    overwrite: Bool = False,
    seed: Int = 0,
    batch_size: Int = 1000,
) -> Dict[str, Array]:
    """
    Run the numerical experiments for Theorem 2.6 in the paper.

    **Arguments:**
        dimension_vals: The dimensions to test.
        noise_std_vals: The noise standard deviations to test.
        prior_std: The prior standard deviation.
        size_cluster_C: The size of the cluster C.
        size_cluster_T: The size of the cluster T.
        n_experiments: The number of experiments to run.
        path_to_output: The path to save the results.
        overwrite: Whether to overwrite the output file if it exists.
        seed: The random seed to use.
        batch_size: The batch size for JAX operations.
    **Returns:**
        results: A dictionary containing the results of the experiments
                and the parameters used.

    The results are also saved in a .npz file in the specified path.
    """
    if os.path.exists(path_to_output) and not overwrite:
        raise ValueError(
            f"Ouput file {path_to_output} exists, but overwrite was set to False"
        )
    else:
        os.makedirs(os.path.dirname(path_to_output), exist_ok=True)

    key = jax.random.key(seed)

    X, Y = np.meshgrid(dimension_vals, noise_std_vals**2, indexing="ij")
    upper_bounds = _compute_upper_bound_main_theorem(
        X, prior_std**2, Y, size_cluster_C, size_cluster_T
    )

    empirical_probs_worst = np.ones((*upper_bounds.shape, n_experiments)) * -1
    empirical_probs_random = np.ones((*upper_bounds.shape, n_experiments)) * -1

    results = {}
    for i in tqdm(range(len(dimension_vals))):
        for j in range(len(noise_std_vals)):
            key, *subkeys = jax.random.split(key, n_experiments + 1)
            subkeys = jnp.array(subkeys)

            empirical_probs_worst[i, j] = jax.lax.map(
                lambda x: _run_experiment_main_theorem_worst(
                    x,
                    size_cluster_C,
                    size_cluster_T,
                    dimension_vals[i],
                    prior_std,
                    noise_std_vals[j],
                ),
                xs=subkeys,
                batch_size=batch_size,
            )

            empirical_probs_random[i, j] = jax.lax.map(
                lambda x: _run_experiment_main_theorem_random(
                    x,
                    size_cluster_C,
                    size_cluster_T,
                    dimension_vals[i],
                    prior_std,
                    noise_std_vals[j],
                ),
                xs=subkeys,
                batch_size=batch_size,
            )
            jax.clear_caches()

            results = {
                "upper_bound": upper_bounds,
                "empirical_probs_worst": empirical_probs_worst,
                "empirical_probs_random": empirical_probs_random,
                "dimension_vals": dimension_vals,
                "noise_std_vals": noise_std_vals,
                "size_cluster_C": size_cluster_C,
                "size_cluster_T": size_cluster_T,
                "n_experiments": n_experiments,
                "prior_std": prior_std,
                "seed": seed,
                "i": i,
                "j": j,
                "key": jax.random.key_data(key),
            }

            jnp.savez(path_to_output, **results)

    return results
