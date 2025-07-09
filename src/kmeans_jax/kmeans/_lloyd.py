from functools import partial
from typing import Tuple

import jax
from jaxtyping import Array, Bool, Float, Int


jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from ._common_functions import (
    assign_clusters,
    compute_loss,
    update_centroids,
)


@jax.jit
def _kmeans_step(
    carry: Tuple[
        Float[Array, "K d"],
        Int[Array, " n"],
        Int[Array, " n"],
        Float[Array, " max_steps"],
        Int,
    ],
    data: Float[Array, "n d"],
) -> Tuple[
    Float[Array, "K d"],
    Int[Array, " n"],
    Int[Array, " n"],
    Float[Array, " max_steps"],
    Int,
]:
    centroids, old_cluster_assignments, _, losses, counter = carry

    cluster_assignments = assign_clusters(centroids, data)
    centroids = update_centroids(data, cluster_assignments, centroids.shape[0])
    # losses = losses.at[counter].set(compute_loss(data, centroids, cluster_assignments))
    losses = compute_loss(data, centroids, cluster_assignments)
    return (centroids, cluster_assignments, old_cluster_assignments, losses, counter + 1)


def _kmeans_stop_condition(
    carry: Tuple[
        Float[Array, "K d"],
        Int[Array, " n"],
        Int[Array, " n"],
        Float[Array, " max_steps"],
        Int,
    ],
    max_steps: Int,
) -> Bool:
    _, cluster_assignments, old_cluster_assignments, losses, counter = carry

    cond1 = jnp.any(cluster_assignments != old_cluster_assignments)
    cond2 = counter <= max_steps

    return cond1 & cond2


def run_kmeans(
    data: Float[Array, "n d"],
    init_centroids: Float[Array, "K d"],
    max_iters: Int = 1000,
) -> Tuple[
    Tuple[Float[Array, "K d"], Int[Array, " n"]],
    Float[Array, " max_steps"],
]:
    """
    Run k-means clustering algorithm from a given initial set of cluster centers.

    **Arguments:**
        data: The data to cluster, shape (n, d).
        init_centroids: The initial centroids, shape (K, d).
        max_iters: The maximum number of iterations to run the algorithm.
    **Returns:**
        centroids: The final centroids, shape (K, d).
        cluster_assignments: The cluster assignments for each data point, shape (n,).
        losses: The loss function value at each iteration.
    """

    losses = 0.0  # jnp.zeros(max_iters)
    counter = 0
    # dummy init
    init_assignments = assign_clusters(init_centroids, data)

    cond_fun = jax.jit(partial(_kmeans_stop_condition, max_steps=max_iters))

    # makes sure the initial assignment does not trigger the stop condition
    carry = (init_centroids, init_assignments, init_assignments - 1, losses, counter)

    @jax.jit
    def run_kmeans_inner(carry):
        return jax.lax.while_loop(
            cond_fun=cond_fun, body_fun=lambda c: _kmeans_step(c, data), init_val=carry
        )

    centroids, assigments, _, losses, counter = run_kmeans_inner(carry)
    # losses = losses[:counter]
    # losses = jax.lax.slice(losses, (0,), (counter,))

    # centroids.block_until_ready()
    return centroids, assigments, losses, counter
