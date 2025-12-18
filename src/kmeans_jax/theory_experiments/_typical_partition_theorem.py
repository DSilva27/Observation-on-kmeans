import os
from functools import partial
from typing import Dict

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Bool, Float, Int, PRNGKeyArray
from tqdm import tqdm

from ..kmeans._common_functions import compute_centroids


def _compute_rho(n_data_points: Int, q_value: Float, noise_variance: Float) -> Float:
    num = (
        noise_variance
        * (jnp.sqrt(n_data_points) * q_value + n_data_points - 2)
        * (n_data_points + jnp.sqrt(n_data_points) * q_value)
        * (jnp.sqrt(n_data_points) * q_value + n_data_points + 2)
        * (
            jnp.sqrt(n_data_points)
            * (noise_variance + 2)
            * (jnp.sqrt(n_data_points) + q_value)
            - 4
        )
    )
    den = (
        n_data_points * noise_variance * (jnp.sqrt(n_data_points) + q_value) ** 2
        + (jnp.sqrt(n_data_points) * q_value + n_data_points - 2) ** 2
    )
    return num / den**2


@partial(jax.vmap, in_axes=(0, None))
@partial(jax.vmap, in_axes=(None, 0))
def _compute_upper_bound(dimension: Int, noise_variance: Float) -> Float:
    return _compute_rho(40, 2.8460497, noise_variance) ** (dimension / 4)


@partial(jax.vmap, in_axes=(0, 0, None, None))  # d
@partial(jax.vmap, in_axes=(0, 0, None, None))  # sigma2
@partial(jax.vmap, in_axes=(0, 0, None, None))  # experiments
def _check_partition_is_valid(
    size_cluster1: Int, size_cluster2: Int, n_data_points: Int, q_value: Float
) -> Bool:
    lower_bound = 0.5 * n_data_points - q_value * jnp.sqrt(0.25 * n_data_points)
    upper_bound = 0.5 * n_data_points + q_value * jnp.sqrt(0.25 * n_data_points)

    b1 = jnp.logical_and(upper_bound > size_cluster1, size_cluster1 > lower_bound)
    b2 = jnp.logical_and(upper_bound > size_cluster2, size_cluster2 > lower_bound)

    return jnp.logical_and(b1, b2)


def _compute_base_noise_std(n_data_points: Int, q_value: Float) -> Float:
    num = jnp.sqrt(n_data_points) * q_value + n_data_points - 2
    den = jnp.sqrt(2 * (jnp.sqrt(n_data_points) * q_value + n_data_points))
    return num / den


@partial(jax.jit, static_argnums=(1, 2))
def _run_experiment_theorem_typical_part(
    key: PRNGKeyArray,
    n_data_points: Int,
    dimension: Int,
    noise_std: Float,
    *,
    idx_data_point: Int = 0,
) -> Int:
    key_centroids, key_labels, key_noise, key_assignment = jax.random.split(key, 4)

    true_centroids = jax.random.normal(key_centroids, shape=(2, dimension))

    true_labels = jax.random.randint(
        key_labels, shape=(n_data_points,), minval=0, maxval=2
    )

    data = (
        jax.random.normal(key_noise, (n_data_points, dimension)) * noise_std
        + true_centroids[true_labels]
    )

    labels = jax.random.randint(
        key_assignment, shape=(n_data_points,), minval=0, maxval=2
    )
    centroids = compute_centroids(data, labels, 2)

    # checking ||x_j - \mu_{\bar{z}(j)}||^2 - ||x_j - \mu_{z(j)}||^2
    # see equation (13) in the paper! <- TODO: check for final version
    dist_to_centroids = jnp.sum(
        (data[idx_data_point][None, ...] - centroids) ** 2, axis=-1
    )

    # makes sure the order of the distance is as in equation (13)
    diff_distance = (-1) ** (labels[idx_data_point]) * (
        dist_to_centroids[1] - dist_to_centroids[0]
    )

    size_cluster_point = jnp.sum(labels == labels[idx_data_point])  # S2
    size_other_cluster = n_data_points - size_cluster_point  # S1

    point_swaps = jax.lax.cond(
        diff_distance < 0,
        lambda _: 1,
        lambda _: 0,
        operand=None,
    )

    return point_swaps, size_other_cluster, size_cluster_point


def run_theorem_typical_part_experiments(
    dimension_vals: Int[Array, " n_dim_vals"],
    beta_vals: Float[Array, " n_noise_std_vals"],
    q_value: Float,
    n_data_points: Int,
    n_experiments: Int,
    path_to_output: str,
    *,
    overwrite: Bool = False,
    seed: Int = 0,
    batch_size: Int = 1000,
) -> Dict[str, Array]:
    """
    Run the experiments for Theorem 2.8.

    **Arguments:**
        dimension_vals: The dimensions to test.
        beta_vals: The noise standard deviations to test.
        q_value: The q value.
        n_data_points: The number of data points.
        n_experiments: The number of experiments to run.
        path_to_output: The path to save the results.
        overwrite: Whether to overwrite the output file if it exists.
        seed: The random seed to use.
        batch_size: The batch size to use for JAX.
    **Returns:**
        results: A dictionary with the results of the experiments.
            and the parameters used.

    The results are also saved in a .npz file in the specified path.
    """
    key = jax.random.key(seed)
    base_noise_std = _compute_base_noise_std(n_data_points, q_value)
    noise_std_vals = beta_vals * base_noise_std

    def _iterate_over_keys(key, d, sigma):
        return jax.lax.map(
            lambda x: _run_experiment_theorem_typical_part(x, n_data_points, d, sigma),
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
    upper_bound = _compute_upper_bound(dimension_vals, noise_std_vals**2)

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
