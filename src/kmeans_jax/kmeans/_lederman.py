from functools import partial
from typing import Tuple

import jax
from jaxtyping import Array, Float, Int


jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from ._common_functions import assign_clusters, compute_loss, update_centroids


# @partial(jax.vmap, in_axes=(0, None, None, 0))
# @partial(jax.vmap, in_axes=(None, 0, 0, 0))
# def weight_distances(assignment, cluster_id, cluster_weight, distance):
#     return jax.lax.cond(
#         assignment == cluster_id,
#         lambda x: jax.lax.cond(
#             cluster_weight <= 1,
#             lambda x: 0.0,
#             lambda x: x * cluster_weight**2 / (cluster_weight - 1) ** 2,
#             distance,
#         ),
#         lambda x: x * cluster_weight**2 / (cluster_weight + 1) ** 2,
#         distance,
#     )
def weight_distance(assignment, cluster_id, cluster_weight, distance):
    # jax.debug.print(
    #     "weight 1: {w}", w=(cluster_weight / (cluster_weight - 1.0)) ** 2
    # )
    # jax.debug.print(
    #     "weight 2: {w}", w=(cluster_weight / (cluster_weight + 1.0)) ** 2
    # )
    return jax.lax.cond(
        assignment == cluster_id,
        lambda x: jax.lax.cond(
            cluster_weight <= 1,
            lambda x: 0.0,
            lambda x: x * (cluster_weight / (cluster_weight - 1.0)) ** 2,
            distance,
        ),
        lambda x: x,  # * (cluster_weight / (cluster_weight + 1.0)) ** 2,
        distance,
    )


def weight_distances(assignments, cluster_ids, cluster_populations, distances):
    return jax.vmap(
        jax.vmap(weight_distance, in_axes=(None, 0, 0, 0)), in_axes=(0, None, None, 0)
    )(assignments, cluster_ids, cluster_populations, distances)


def update_assignments(centroids, assignments, data):
    cluster_populations = jnp.bincount(assignments, length=centroids.shape[0])
    # jax.debug.print(
    #     "populations: {a}", a=cluster_populations
    # )
    distances = jnp.sum((data[:, None, :] - centroids[None, :, :]) ** 2, axis=-1)
    distances = weight_distances(
        assignments, jnp.arange(centroids.shape[0]), cluster_populations, distances
    )
    return jnp.argmin(distances, axis=-1)


def _lederman_step(carry, data):
    centroids, old_assignments, old_old_assignments, _, loss, counter = carry

    assignments = update_assignments(centroids, old_assignments, data)

    centroids = update_centroids(data, assignments, centroids.shape[0])
    loss = compute_loss(data, centroids, assignments)

    return (
        centroids,
        assignments,
        old_assignments,
        old_old_assignments,
        loss,
        counter + 1,
    )


def _lederman_stop_condition(carry, max_steps):
    (
        centroids,
        assignments,
        old_assignments,
        old_old_assignments,
        loss,
        counter,
    ) = carry

    cond1 = jnp.any(assignments != old_assignments)
    cond2 = jnp.any(assignments != old_old_assignments)
    cond3 = counter <= max_steps

    return cond1 & cond2 & cond3


def run_lederman_kmeans(
    data: Float[Array, "n d"],
    init_centroids: Float[Array, "K d"],
    max_iters: Int = 1000,
) -> Tuple[
    Tuple[Float[Array, "K d"], Int[Array, " n"]],
    Float[Array, " max_steps"],
]:
    loss = 0.0  # jnp.zeros(max_iters)
    counter = 0

    init_assignments = assign_clusters(init_centroids, data)
    init_centroids = update_centroids(data, init_assignments, init_centroids.shape[0])

    cond_fun = jax.jit(partial(_lederman_stop_condition, max_steps=max_iters))

    # makes sure the initial assignment does not trigger the stop condition
    carry = (
        init_centroids,
        init_assignments,
        init_assignments - 1,
        init_assignments - 2,
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

    centroids, assignments, _, _, loss, counter = run_lederman_inner(carry)
    return centroids, assignments, loss, counter
