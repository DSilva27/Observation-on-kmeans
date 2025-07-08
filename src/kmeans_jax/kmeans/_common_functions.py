from functools import partial

import jax
from jaxtyping import Array, Float, Int


jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp


@jax.jit
def compute_loss(
    data: Float[Array, "n d"],
    centroids: Float[Array, "K d"],
    cluster_assignments: Int[Array, " n"],
) -> Float:
    """
    Compute the loss function for k-means clustering.

    **Arguments:**
        data: The data to cluster, shape (n, d).
        centroids: The centroids of the clusters, shape (K, d).
        cluster_assignments: The cluster assignments for each data point, shape (n,).
    **Returns:**
        loss: The loss function value.
    """
    return jnp.sum(jnp.abs(data - centroids[cluster_assignments]) ** 2)


@jax.jit
def assign_clusters(
    centroids: Float[Array, "K d"], data: Float[Array, "n d"]
) -> Int[Array, " n"]:
    """
    Assign each data point to the nearest centroid.

    **Arguments:**
        centroids: The centroids of the clusters, shape (K, d).
        data: The data to cluster, shape (n, d).
    **Returns:**
        cluster_assignments: The cluster assignments for each data point, shape (n,).
    """
    distances = jnp.linalg.norm(data[:, None, :] - centroids[None, :, :], axis=-1)
    return jnp.argmin(distances, axis=1)


@partial(jax.jit, static_argnums=(2,))
def update_centroids(
    data: Float[Array, "n d"], cluster_assignments: Int[Array, " n"], num_clusters: Int
) -> Float[Array, "K d"]:
    """
    Update the centroids of the clusters, given the data and cluster assignments.

    **Arguments:**
        data: The data to cluster, shape (n, d).
        cluster_assignments: The cluster assignments for each data point, shape (n,).
        num_clusters: The number of clusters.
    **Returns:**
        centroids: The updated centroids, shape (K, d).
    """

    def compute_centroid(cluster):
        num = jnp.sum(
            data,
            axis=0,
            where=jnp.where(cluster_assignments == cluster, True, False)[:, None],
        )
        den = jnp.sum(jnp.where(cluster_assignments == cluster, True, False)) + 1e-16
        return num / den

    return jax.vmap(compute_centroid)(jnp.arange(num_clusters))
