import os

import jax
import numpy as np
from tqdm import tqdm


jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from ..kmeans import (
    assign_clusters,
    update_centroids,
)
from ._main_theorem import _compute_upper_bound_main_theorem


def _run_experiment_union_bound_cor(
    key, cluster_size, dimension, prior_variance, noise_variance
):
    key1, key2, key_data1, key_data2, key_init = jax.random.split(key, 5)
    true_mu_C = jax.random.normal(key1, shape=(dimension,)) * jnp.sqrt(prior_variance)
    true_mu_T = jax.random.normal(key2, shape=(dimension,)) * jnp.sqrt(prior_variance)

    x_C = true_mu_C[None, ...] + jax.random.normal(
        key_data1, shape=(cluster_size, dimension)
    ) * jnp.sqrt(noise_variance)
    x_T = true_mu_T[None, ...] + jax.random.normal(
        key_data2, shape=(cluster_size, dimension)
    ) * jnp.sqrt(noise_variance)

    true_assignments = jnp.concatenate(
        (jnp.zeros(cluster_size), jnp.ones(cluster_size))
    ).astype(int)

    data = jnp.concatenate([x_C, x_T])
    assignments1 = jax.random.choice(
        key_init, true_assignments, shape=(data.shape[0],), replace=False
    )
    centroids = update_centroids(data, assignments1, 2)
    assignments2 = assign_clusters(centroids, data)

    n_points_swapped = jnp.sum(assignments1 != assignments2)
    return n_points_swapped


def run_union_bound_cor_experiments(
    dimension_vals,
    noise_std_vals,
    prior_std,
    cluster_size,
    n_experiments,
    path_to_output,
    *,
    overwrite=False,
    seed=0,
    batch_size=1000,
):
    if os.path.exists(path_to_output) and not overwrite:
        raise ValueError(
            f"Ouput file {path_to_output} exists, but overwrite was set to False"
        )
    else:
        os.makedirs(os.path.dirname(path_to_output), exist_ok=True)

    key = jax.random.key(seed)

    X, Y = np.meshgrid(dimension_vals, noise_std_vals**2, indexing="ij")
    upper_bounds = _compute_upper_bound_main_theorem(
        X, prior_std**2, Y, cluster_size, cluster_size
    )

    empirical_probs = np.ones((*upper_bounds.shape, n_experiments)) * -1.0

    results = {}
    for i in tqdm(range(len(dimension_vals))):
        for j in range(len(noise_std_vals)):
            key, *subkeys = jax.random.split(key, n_experiments + 1)
            subkeys = jnp.array(subkeys)

            empirical_probs[i, j] = jax.lax.map(
                lambda x: _run_experiment_union_bound_cor(
                    x, cluster_size, dimension_vals[i], prior_std, noise_std_vals[j]
                ),
                xs=subkeys,
                batch_size=batch_size,
            )

            jax.clear_caches()

            results = {
                "upper_bound": upper_bounds,
                "empirical_probs": empirical_probs,
                "dimension_vals": dimension_vals,
                "noise_std_vals": noise_std_vals,
                "cluster_size": cluster_size,
                "n_experiments": n_experiments,
                "prior_std": prior_std,
                "seed": seed,
                "key": jax.random.key_data(key),
            }

            jnp.savez(path_to_output, **results)
    return results



