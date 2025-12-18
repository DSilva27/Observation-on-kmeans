from functools import partial
from typing import Tuple

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int, PRNGKeyArray

from ._common_functions import compute_centroids


@partial(jax.jit, static_argnums=(1,))
def kmeans_plusplus_init(
    data: Float[Array, "n d"], K: Int, key: PRNGKeyArray
) -> Tuple[Float[Array, "K d"], Int[Array, " K"]]:
    """
    K-means++ initialization for k-means clustering.

    **Arguments:**
        data: The data to cluster, shape (n, d).
        K: The number of clusters.
        key: A JAX random key.
    **Returns:**
        centroids: The initial centroids, shape (K, d).
        indices: The indices of the data points used as initial centroids, shape (K,).
    """
    key, subkey = jax.random.split(key)
    init_centroids = jnp.zeros((K, data.shape[-1]), dtype=data.dtype)
    indices = jnp.zeros((K,), dtype=int)

    first_index = jax.random.choice(subkey, data.shape[0])
    indices = indices.at[0].set(first_index)
    init_centroids = init_centroids.at[0].set(data[indices[0]])

    def body_fun(i, val):
        centroids, indices, key = val
        mask = jnp.arange(K) < i
        valid_centroids = jnp.where(mask[:, None], centroids, jnp.inf)

        distances = jnp.linalg.norm(data[:, None, :] - valid_centroids, axis=-1)
        min_distances = jnp.min(distances, axis=1) ** 2

        key, subkey = jax.random.split(key)
        new_index = jax.random.choice(
            subkey, data.shape[0], p=min_distances / jnp.sum(min_distances)
        )
        indices = indices.at[i].set(new_index)
        centroids = centroids.at[i].set(data[new_index])
        return (centroids, indices, key)

    centroids, indices, _ = jax.lax.fori_loop(
        1, K, body_fun, (init_centroids, indices, key)
    )
    return centroids, indices


def kmeans_random_init(
    data: Float[Array, "n d"], K: Int, key: PRNGKeyArray
) -> Tuple[Float[Array, "K d"], Int[Array, " K"]]:
    """
    Random initialization for k-means clustering.

    **Arguments:**
        data: The data to cluster, shape (n, d).
        K: The number of clusters.
        key: A JAX random key.
    **Returns:**
        centroids: The initial centroids, shape (K, d).
        indices: The indices of the data points used as initial centroids, shape (K,).
    """
    indices = jax.random.choice(key, data.shape[0], (K,), replace=False)
    return data[indices], indices


def kmeans_init_from_random_partition(
    data: Float[Array, "n d"],
    K: Int,
    key: PRNGKeyArray,
    *,
    labels: Int[Array, " n"] = None,
) -> Tuple[Float[Array, "K d"], Int[Array, " n"]]:
    """
    Random partition initialization for k-means clustering.

    **Arguments:**
        data: The data to cluster, shape (n, d).
        K: The number of clusters.
        key: A JAX random key.
        labels: Optional true labels for the data points, shape (n,). If provided,
                the initial partition will be based on these labels.
    **Returns:**
        partition: The initial partition of the data points into clusters, shape (n,).
        centroids: The initial centroids, shape (K, d).
    """
    if labels is not None:
        partition = jax.random.choice(key, labels, shape=(data.shape[0],), replace=False)
    else:
        partition = jax.random.choice(key, K, (data.shape[0],))

    centroids = compute_centroids(data, partition, K)
    return centroids, partition
