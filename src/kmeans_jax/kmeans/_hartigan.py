from functools import partial
from typing import Tuple

import jax
from jaxtyping import Array, Bool, Float, Int


jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from ._common_functions import assign_clusters, compute_loss, update_centroids


def weight_distance(assignment, cluster_id, cluster_weight, distance):
    return jax.lax.cond(
        assignment == cluster_id,
        lambda x: x * cluster_weight / (cluster_weight - 1),
        lambda x: x * cluster_weight / (cluster_weight + 1),
        distance,
    )


def assign_dp_to_cluster_hartigan(centroids, assignments, assignment_point, data_point):
    cluster_weights = jnp.bincount(assignments, length=centroids.shape[0])

    distances = jnp.sum((data_point[None, ...] - centroids) ** 2, axis=-1)
    distances = jax.vmap(weight_distance, in_axes=(None, 0, 0, 0))(
        assignment_point, jnp.arange(centroids.shape[0]), cluster_weights, distances
    )
    return jnp.argmin(distances)


def inner_loop_hartigan(assignments, centroids, data):
    def body_fun(i, val):
        assignments, centroids = val

        assignment = assign_dp_to_cluster_hartigan(
            centroids, assignments, assignments[i], data[i]
        )

        pred = assignments[i] != assignment
        assignments = assignments.at[i].set(assignment)

        centroids = jax.lax.cond(
            pred,
            lambda x: update_centroids(data, assignments, centroids.shape[0]),
            lambda x: x,
            centroids,
        )

        return (assignments, centroids)

    return jax.lax.fori_loop(0, data.shape[0], body_fun, (assignments, centroids))


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
    centroids, old_assignments, _, losses, counter = carry

    assignments, centroids = inner_loop_hartigan(old_assignments.copy(), centroids, data)
    # losses = losses.at[counter].set(compute_loss(data, centroids, assignments))
    losses = compute_loss(data, centroids, assignments)
    return (centroids, assignments, old_assignments, losses, counter + 1)


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
    _, assignments, old_assignments, _, counter = carry

    cond1 = jnp.any(assignments != old_assignments)
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
    losses = 0.0  # jnp.zeros(max_iters)
    counter = 0

    init_assignments = assign_clusters(init_centroids, data)

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
    # losses = losses[:counter]
    # centroids.block_until_ready()
    return centroids, assigments, losses, counter


############################ Batched Hartigan ############################
def weight_distances(assignments, cluster_ids, cluster_weights, distances):
    return jax.vmap(
        jax.vmap(weight_distance, in_axes=(None, 0, 0, 0)), in_axes=(0, None, None, 0)
    )(assignments, cluster_ids, cluster_weights, distances)


def assign_dp_to_cluster_batched_hartigan(centroids, assignments, data):
    cluster_weights = jnp.bincount(assignments, length=centroids.shape[0])
    cluster_weights = jnp.clip(cluster_weights, min=2, max=None)

    distances = jnp.sum((data[:, None, :] - centroids[None, :, :]) ** 2, axis=-1)
    distances = weight_distances(
        assignments, jnp.arange(centroids.shape[0]), cluster_weights, distances
    )
    return jnp.argmin(distances, axis=-1)


def _batched_hartigan_step(carry, data):
    centroids, old_assignments, old_old_assignments, _, losses, counter = carry

    assignments = assign_dp_to_cluster_batched_hartigan(centroids, old_assignments, data)
    centroids = update_centroids(data, assignments, centroids.shape[0])
    # losses = losses.at[counter].set(compute_loss(data, centroids, assignments))
    losses = compute_loss(data, centroids, assignments)
    return (
        centroids,
        assignments,
        old_assignments,
        old_old_assignments,
        losses,
        counter + 1,
    )


def _batched_hartigan_stop_condition(carry, max_steps):
    (
        centroids,
        assignments,
        old_assignments,
        old_old_assignments,
        losses,
        counter,
    ) = carry

    # loss_prev_prev = jax.lax.cond(
    #     counter > 2, lambda x: losses[x - 3], lambda x: jnp.inf, counter
    # )
    # loss_prev = losses[counter - 2]
    # loss = losses[counter - 1]
    # loss_diff_1 = jnp.abs(loss - loss_prev)
    # loss_diff_2 = jnp.abs(loss_prev - loss_prev_prev)

    cond1 = jnp.any(assignments != old_assignments)
    cond2 = jnp.any(assignments != old_old_assignments)
    cond3 = counter <= max_steps
    cond4 = jnp.all(jnp.bincount(assignments, length=centroids.shape[0]) > 1)
    return cond1 & cond2 & cond3 & cond4


def run_batched_hartigan_kmeans(
    data: Float[Array, "n d"],
    init_centroids: Float[Array, "K d"],
    max_iters: Int = 1000,
) -> Tuple[
    Tuple[Float[Array, "K d"], Int[Array, " n"]],
    Float[Array, " max_steps"],
]:
    losses = 0.0  # jnp.zeros(max_iters)
    counter = 0

    init_assignments = assign_clusters(init_centroids, data)
    cond_fun = jax.jit(partial(_batched_hartigan_stop_condition, max_steps=max_iters))

    # makes sure the initial assignment does not trigger the stop condition
    carry = (
        init_centroids,
        init_assignments,
        init_assignments - 1,
        init_assignments - 2,
        losses,
        counter,
    )

    @jax.jit
    def run_batched_hartigan_inner(carry):
        return jax.lax.while_loop(
            cond_fun=cond_fun,
            body_fun=lambda c: _batched_hartigan_step(c, data),
            init_val=carry,
        )

    centroids, assigments, _, _, losses, counter = run_batched_hartigan_inner(carry)
    # losses = losses[:counter]
    # centroids.block_until_ready()
    return centroids, assigments, losses, counter
