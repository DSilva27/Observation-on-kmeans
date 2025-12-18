from functools import partial

import jax


import jax.numpy as jnp
from jaxtyping import Array, Float, Int


@jax.jit
def compute_loss(
    data: Float[Array, "n d"],
    masks: Int[Array, "n d"],
    centroids: Float[Array, "K d"],
    labels: Int[Array, " n"],
) -> Float:
    """
    Compute the loss function for k-means clustering.

    **Arguments:**
        data: The data to cluster, shape (n, d).
        centroids: The centroids of the clusters, shape (K, d).
        labels: The cluster labels for each data point, shape (n,).
    **Returns:**
        loss: The loss function value.
    """
    return jnp.sum(jnp.abs(data - masks * centroids[labels]) ** 2)


@jax.jit
def assign_clusters(
    centroids: Float[Array, "K d"], data: Float[Array, "n d"], masks: Int[Array, "n d"]
) -> Int[Array, " n"]:
    """
    Assign each data point to the nearest centroid.

    **Arguments:**
        centroids: The centroids of the clusters, shape (K, d).
        data: The data to cluster, shape (n, d).
    **Returns:**
        labels: The cluster labels for each data point, shape (n,).
    """
    distances = jnp.linalg.norm(
        data[:, None, :] - masks[:, None, :] * centroids[None, :, :], axis=-1
    )
    return jnp.argmin(distances, axis=1)


@partial(jax.jit, static_argnums=(3,))
def compute_centroids(
    data: Float[Array, "n d"],
    masks: Int[Array, "n d"],
    labels: Int[Array, " n"],
    n_clusters: Int,
) -> Float[Array, "K d"]:
    """
    Update the centroids of the clusters, given the data and cluster labels.

    **Arguments:**
        data: The data to cluster, shape (n, d).
        labels: The cluster labels for each data point, shape (n,).
        n_clusters: The number of clusters.
    **Returns:**
        centroids: The updated centroids, shape (K, d).
    """

    masked_data = data * masks

    def _compute_centroid(cluster):
        num = jnp.sum(
            masked_data,
            axis=0,
            where=jnp.where(labels == cluster, True, False)[:, None],
        )
        den = jnp.sum(
            masks,
            axis=0,
            where=jnp.where(labels == cluster, True, False)[:, None],
        )
        return jnp.where(jnp.isclose(den, 0.0), 0.0, num / den)

    return jax.vmap(_compute_centroid)(jnp.arange(n_clusters))


# sum_i A_i x_i / (sum_i A_i)
