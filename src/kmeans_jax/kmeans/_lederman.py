from functools import partial
from typing import Tuple

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int

from ._common_functions import assign_clusters, compute_loss, compute_centroids



def weight_distance(assignment, cluster_id, cluster_weight, distance):
    return jax.lax.cond(
        assignment == cluster_id,
        lambda x: jax.lax.cond(
            cluster_weight <= 1,
            lambda x: 0.0,
            lambda x: x * (cluster_weight / (cluster_weight - 1.0)) ** 2,
            distance,
        ),
        lambda x: x,
        distance,
    )


def weight_distances(labels, cluster_ids, cluster_populations, distances):
    return jax.vmap(
        jax.vmap(weight_distance, in_axes=(None, 0, 0, 0)), in_axes=(0, None, None, 0)
    )(labels, cluster_ids, cluster_populations, distances)


def _update_labels(centroids, labels, data):
    cluster_populations = jnp.bincount(labels, length=centroids.shape[0])
    # jax.debug.print(
    #     "populations: {a}", a=cluster_populations
    # )
    distances = jnp.sum((data[:, None, :] - centroids[None, :, :]) ** 2, axis=-1)
    distances = weight_distances(
        labels, jnp.arange(centroids.shape[0]), cluster_populations, distances
    )
    return jnp.argmin(distances, axis=-1)


def _lederman_step(carry, data):
    centroids, old_labels, old_old_labels, _, loss, counter = carry

    labels = _update_labels(centroids, old_labels, data)

    centroids = compute_centroids(data, labels, centroids.shape[0])
    loss = compute_loss(data, centroids, labels)

    return (
        centroids,
        labels,
        old_labels,
        old_old_labels,
        loss,
        counter + 1,
    )


def _lederman_stop_condition(carry, max_steps):
    (
        centroids,
        labels,
        old_labels,
        old_old_labels,
        loss,
        counter,
    ) = carry

    cond1 = jnp.any(labels != old_labels)
    cond2 = jnp.any(labels != old_old_labels)
    cond3 = counter <= max_steps

    return cond1 & cond2 & cond3


def run_lederman_kmeans(
    data: Float[Array, "n d"],
    init_centroids: Float[Array, "K d"],
    max_iters: Int = 1000,
) -> Tuple[
    Float[Array, "K d"],
    Int[Array, " n"],
    Float,
    Int,
]:
    loss = 0.0  # jnp.zeros(max_iters)
    counter = 0

    init_labels = assign_clusters(init_centroids, data)
    init_centroids = compute_centroids(data, init_labels, init_centroids.shape[0])

    cond_fun = jax.jit(partial(_lederman_stop_condition, max_steps=max_iters))

    # makes sure the initial assignment does not trigger the stop condition
    carry = (
        init_centroids,
        init_labels,
        init_labels - 1,
        init_labels - 2,
        loss,
        counter,
    )

    @jax.jit
    def run_lederman_inner(carry):
        return jax.lax.while_loop(
            cond_fun=cond_fun,
            body_fun=lambda c: _lederman_step(c, data),
            init_val=carry,
        )

    centroids, labels, _, _, loss, counter = run_lederman_inner(carry)
    return centroids, labels, loss, counter
