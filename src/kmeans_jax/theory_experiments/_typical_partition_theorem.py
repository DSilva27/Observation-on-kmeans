import os
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm

from ..kmeans import update_centroids


def compute_rho(n, q, s2):
    num = (
        s2
        * (jnp.sqrt(n) * q + n - 2)
        * (n + jnp.sqrt(n) * q)
        * (jnp.sqrt(n) * q + n + 2)
        * (jnp.sqrt(n) * (s2 + 2) * (jnp.sqrt(n) + q) - 4)
    )
    den = n * s2 * (jnp.sqrt(n) + q) ** 2 + (jnp.sqrt(n) * q + n - 2) ** 2
    return num / den**2


@partial(jax.vmap, in_axes=(0, None))
@partial(jax.vmap, in_axes=(None, 0))
def compute_upper_bound(d, s2):
    return compute_rho(40, 2.8460497, s2) ** (d / 4)


@partial(jax.vmap, in_axes=(0, 0, None, None))  # d
@partial(jax.vmap, in_axes=(0, 0, None, None))  # sigma2
@partial(jax.vmap, in_axes=(0, 0, None, None))  # experiments
def check_partition_is_valid(S1, S2, n, q):
    lower_bound = 0.5 * n - q * jnp.sqrt(0.25 * n)
    upper_bound = 0.5 * n + q * jnp.sqrt(0.25 * n)

    b1 = jnp.logical_and(upper_bound > S1, S1 > lower_bound)
    b2 = jnp.logical_and(upper_bound > S2, S2 > lower_bound)

    return jnp.logical_and(b1, b2)

def compute_base_noise_std(n, q):
    num = jnp.sqrt(n) * q + n - 2
    den = jnp.sqrt(2 * (jnp.sqrt(n) * q + n))
    return num / den


@partial(jax.jit, static_argnums=(1, 2))
def run_experiment_theorem_typical_part(
    key, n_data_points, dimension, noise_std, *, idx_data_point=0
):
    key_centroids, key_labels, key_noise, key_assignment = jax.random.split(key, 4)

    true_centroids = jax.random.normal(key_centroids, shape=(2, dimension))

    true_labels = jax.random.randint(
        key_labels, shape=(n_data_points,), minval=0, maxval=2
    )

    data = (
        jax.random.normal(key_noise, (n_data_points, dimension)) * noise_std
        + true_centroids[true_labels]
    )

    assignments = jax.random.randint(
        key_assignment, shape=(n_data_points,), minval=0, maxval=2
    )
    centroids = update_centroids(data, assignments, 2)

    # checking ||x_j - \mu_{\bar{z}(j)}||^2 - ||x_j - \mu_{z(j)}||^2
    # see equation (13) in the paper! <- TODO: check for final version
    dist_to_centroids = jnp.sum(
        (data[idx_data_point][None, ...] - centroids) ** 2, axis=-1
    )

    # makes sure the order of the distance is as in equation (13)
    diff_distance = (-1) ** (assignments[idx_data_point]) * (
        dist_to_centroids[1] - dist_to_centroids[0]
    )

    size_cluster_point = jnp.sum(assignments == assignments[idx_data_point])  # S2
    size_other_cluster = n_data_points - size_cluster_point  # S1

    if diff_distance < 0:
        point_swaps = 1
    else:
        point_swaps = 0

    return point_swaps, size_other_cluster, size_cluster_point


def run_theorem_typical_part_experiments(
    dimension_vals,
    beta_vals,
    q_value,
    n_data_points,
    n_experiments,
    path_to_output,
    *,
    overwrite=False,
    seed=0,
    batch_size=1000,
):
    key = jax.random.key(seed)
    base_noise_std = compute_base_noise_std(n_data_points, q_value)
    noise_std_vals = beta_vals * base_noise_std

    def _iterate_over_keys(key, d, sigma):
        return jax.lax.map(
            lambda x: run_experiment_theorem_typical_part(x, n_data_points, d, sigma),
            xs=jax.random.split(key, n_experiments),
            batch_size=batch_size,
        )

    def _iterate_over_sigma(key, d, sigmas):
        return jax.lax.map(
            lambda x: _iterate_over_keys(x[0], d, x[1]),
            xs=(jax.random.split(key, sigmas.shape[0]), sigmas),
        )

    if os.path.exists(path_to_output) and not overwrite:
        raise ValueError(
            f"Ouput file {path_to_output} exists, but overwrite was set to False"
        )
    else:
        os.makedirs(os.path.dirname(path_to_output), exist_ok=True)

    shape_parameter_space = (len(dimension_vals), len(noise_std_vals))
    empirical_probs = np.ones((*shape_parameter_space, n_experiments)) * -1.0
    cluster_sizes = np.ones((*shape_parameter_space, n_experiments, 2), dtype=int) * -1
    upper_bound = compute_upper_bound(dimension_vals, noise_std_vals ** 2)

    results = {}

    for i in tqdm(range(len(dimension_vals))):
        key, subkey = jax.random.split(key)
        outputs = _iterate_over_sigma(subkey, dimension_vals[i], noise_std_vals)
        empirical_probs[i] = outputs[0]
        cluster_sizes[i, :, :, 0] = outputs[1]
        cluster_sizes[i, :, :, 1] = outputs[2]

        jax.clear_caches()
        results = {
            "empirical_probs": empirical_probs,
            "upper_bound": upper_bound,
            "cluster_sizes": cluster_sizes,
            "dimension_vals": dimension_vals,
            "noise_std_vals": noise_std_vals,
            "beta_vals": beta_vals,
            "base_noise_std": base_noise_std,
            "q_value": q_value,
            "n_data_points": n_data_points,
            "n_experiments": n_experiments,
            "seed": seed,
            "i": i,
            "key": jax.random.key_data(key),
        }

        jnp.savez(path_to_output, **results)
    return results