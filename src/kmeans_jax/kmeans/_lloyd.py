from functools import partial
from typing import Tuple

import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int

from ._common_functions import (
    assign_clusters,
    compute_centroids,
    compute_loss,
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
    centroids, old_labels, _, loss, counter = carry

    labels = assign_clusters(centroids, data)
    centroids = compute_centroids(data, labels, centroids.shape[0])
    # loss = loss.at[counter].set(compute_loss(data, centroids, labels))
    loss = compute_loss(data, centroids, labels)
    return (centroids, labels, old_labels, loss, counter + 1)


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
    _, labels, old_labels, loss, counter = carry

    cond1 = jnp.any(labels != old_labels)
    cond2 = counter <= max_steps

    return cond1 & cond2


def run_lloyd_kmeans(
    data: Float[Array, "n d"],
    init_centroids: Float[Array, "K d"],
    max_iters: Int = 1000,
) -> Tuple[Float[Array, "K d"], Int[Array, " n"], Float, Int]:
    """
    Run Lloyd's k-means clustering algorithm from a given initial set of cluster centers.

    **Arguments:**
        data: The data to cluster, shape (n, d).
        init_centroids: The initial centroids, shape (K, d).
        max_iters: The maximum number of iterations to run the algorithm.
    **Returns:**
        centroids: The final centroids, shape (K, d).
        labels: The cluster labels for each data point, shape (n,).
        loss: The loss function value at each iteration.
        counter: The number of iterations performed.
    """

    loss = 0.0  # jnp.zeros(max_iters)
    counter = 0
    # dummy init
    init_labels = assign_clusters(init_centroids, data)
    init_centroids = compute_centroids(data, init_labels, init_centroids.shape[0])

    cond_fun = jax.jit(partial(_kmeans_stop_condition, max_steps=max_iters))

    # makes sure the initial assignment does not trigger the stop condition
    carry = (init_centroids, init_labels, init_labels - 1, loss, counter)

    @jax.jit
    def run_lloyd_kmeans_inner(carry):
        return jax.lax.while_loop(
            cond_fun=cond_fun, body_fun=lambda c: _kmeans_step(c, data), init_val=carry
        )

    centroids, assigments, _, loss, counter = run_lloyd_kmeans_inner(carry)
    # loss = loss[:counter]
    # loss = jax.lax.slice(loss, (0,), (counter,))

    # centroids.block_until_ready()
    return centroids, assigments, loss, counter
