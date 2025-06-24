from functools import partial
from typing import Tuple

import jax
from jaxtyping import Array, Bool, Float, Int


jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from .kmeans import assign_clusters, compute_loss, update_centroids


def weight_distance(assignment, cluster_id, cluster_weight, distance):
    return jax.lax.cond(
        assignment == cluster_id,
        lambda x: x * cluster_weight / (cluster_weight - 1),
        lambda x: x * cluster_weight / (cluster_weight + 1),
        distance,
    )


def assign_dp_to_cluster_hartigan(
    centroids, cluster_assignments, assignment_point, data_point
):
    cluster_weights = jnp.bincount(cluster_assignments, length=centroids.shape[0])

    distances = jnp.linalg.norm(data_point[None, ...] - centroids, axis=-1) ** 2
    distances = jax.vmap(weight_distance, in_axes=(None, 0, 0, 0))(
        assignment_point, jnp.arange(centroids.shape[0]), cluster_weights, distances
    )
    return jnp.argmin(distances)


def inner_loop_hartigan(cluster_assignments, centroids, data):
    def body_fun(i, val):
        cluster_assignments, centroids = val
        # assignment = assign_dp_to_cluster_modified(
        #     centroids_num, centroids_den, cluster_assignments[i], data[i]
        # )
        assignment = assign_dp_to_cluster_hartigan(
            centroids, cluster_assignments, cluster_assignments[i], data[i]
        )

        pred = cluster_assignments[i] != assignment
        cluster_assignments = cluster_assignments.at[i].set(assignment)

        centroids = jax.lax.cond(
            pred,
            lambda x: update_centroids(data, cluster_assignments, centroids.shape[0]),
            lambda x: x,
            centroids,
        )

        return (cluster_assignments, centroids)

    return jax.lax.fori_loop(0, data.shape[0], body_fun, (cluster_assignments, centroids))


def _hartigan_kmeans_step(
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

    cluster_assignments, centroids = inner_loop_hartigan(
        old_cluster_assignments.copy(), centroids, data
    )
    losses = losses.at[counter].set(compute_loss(data, centroids, cluster_assignments))
    return (centroids, cluster_assignments, old_cluster_assignments, losses, counter + 1)


def _hart_kmeans_stop_condition(
    carry: Tuple[
        Float[Array, "K d"],
        Int[Array, " n"],
        Int[Array, " n"],
        Float[Array, " max_steps"],
        Int,
    ],
    max_steps: Int,
) -> Bool:
    _, cluster_assignments, old_cluster_assignments, _, counter = carry

    cond1 = jnp.any(cluster_assignments != old_cluster_assignments)
    cond2 = counter <= max_steps

    return cond1 & cond2


def run_hartigan_kmeans(
    data: Float[Array, "n d"],
    init_centroids: Float[Array, "K d"],
    max_iters: Int = 1000,
) -> Tuple[
    Tuple[Float[Array, "K d"], Int[Array, " n"]],
    Float[Array, " max_steps"],
]:
    losses = jnp.zeros(max_iters)
    counter = 0
    # dummy init
    init_assignments = jnp.ones(data.shape[0], dtype=int) * -1

    cond_fun = jax.jit(partial(_hart_kmeans_stop_condition, max_steps=max_iters))

    # makes sure the initial assignment does not trigger the stop condition
    carry = (init_centroids, init_assignments, init_assignments - 1, losses, counter)

    @jax.jit
    def run_batched_hartigan_inner(carry):
        return jax.lax.while_loop(
            cond_fun=cond_fun,
            body_fun=lambda c: _hartigan_kmeans_step(c, data),
            init_val=carry,
        )

    centroids, assigments, _, losses, counter = run_batched_hartigan_inner(carry)
    losses = losses[:counter]

    centroids.block_until_ready()
    return (centroids, assigments), losses


### Batched Hartigan


def weight_distances(cluster_assignments, cluster_ids, cluster_weights, distances):
    return jax.vmap(
        jax.vmap(weight_distance, in_axes=(None, 0, 0, 0)), in_axes=(0, None, None, 0)
    )(cluster_assignments, cluster_ids, cluster_weights, distances)


def assign_dp_to_cluster_batched_hartigan(centroids, cluster_assignments, data):
    cluster_weights = jnp.bincount(cluster_assignments, length=centroids.shape[0])
    distances = jnp.linalg.norm(data[:, None, :] - centroids[None, :, :], axis=-1) ** 2
    distances = weight_distances(
        cluster_assignments, jnp.arange(centroids.shape[0]), cluster_weights, distances
    )
    return jnp.argmin(distances, axis=-1)


def _batched_hartigan_step(carry, data):
    centroids, old_cluster_assignments, _, losses, counter = carry

    cluster_assignments = assign_dp_to_cluster_batched_hartigan(
        centroids, old_cluster_assignments, data
    )
    centroids = update_centroids(data, cluster_assignments, centroids.shape[0])
    losses = losses.at[counter].set(compute_loss(data, centroids, cluster_assignments))
    return (centroids, cluster_assignments, old_cluster_assignments, losses, counter + 1)


def _batched_hartigan_stop_condition(carry, max_steps):
    _, cluster_assignments, old_cluster_assignments, losses, counter = carry

    # loss_prev_prev = jax.lax.cond(
    #     counter > 2, lambda x: losses[x - 3], lambda x: jnp.inf, counter
    # )
    # loss_prev = losses[counter - 2]
    # loss = losses[counter - 1]
    # loss_diff_1 = jnp.abs(loss - loss_prev)
    # loss_diff_2 = jnp.abs(loss_prev - loss_prev_prev)

    cond1 = jnp.any(cluster_assignments != old_cluster_assignments)
    # cond2 = jnp.abs(loss_diff_1 - loss_diff_2) / loss_diff_1 > 1e-6
    cond3 = counter <= max_steps
    # return cond1 & cond2 & cond3
    return cond1 & cond3


def run_batched_hartigan_kmeans(
    data: Float[Array, "n d"],
    init_centroids: Float[Array, "K d"],
    max_iters: Int = 1000,
) -> Tuple[
    Tuple[Float[Array, "K d"], Int[Array, " n"]],
    Float[Array, " max_steps"],
]:
    losses = jnp.zeros(max_iters)
    counter = 0
    # dummy init
    init_assignments = assign_clusters(init_centroids, data)
    cond_fun = jax.jit(partial(_batched_hartigan_stop_condition, max_steps=max_iters))

    # makes sure the initial assignment does not trigger the stop condition
    carry = (init_centroids, init_assignments, init_assignments - 1, losses, counter)

    @jax.jit
    def run_batched_hartigan_inner(carry):
        return jax.lax.while_loop(
            cond_fun=cond_fun,
            body_fun=lambda c: _batched_hartigan_step(c, data),
            init_val=carry,
        )

    centroids, assigments, _, losses, counter = run_batched_hartigan_inner(carry)
    losses = losses[:counter]

    centroids.block_until_ready()
    return (centroids, assigments), losses
