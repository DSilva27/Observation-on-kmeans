from functools import partial
from typing import Tuple

import jax

# jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int, PRNGKeyArray

from ._common_functions import assign_clusters, compute_loss, update_centroids


def weight_distance(assignment, cluster_id, cluster_weight, distance):
    return jax.lax.cond(
        assignment == cluster_id,
        lambda x: jax.lax.cond(
            cluster_weight <= 1,
            lambda x: 0.0,
            lambda x: x * cluster_weight / (cluster_weight - 1),
            distance,
        ),
        lambda x: x * cluster_weight / (cluster_weight + 1),
        distance,
    )


def assign_dp_to_cluster_hartigan(centroids, assignments, assignment_point, data_point):
    cluster_populations = jnp.bincount(assignments, length=centroids.shape[0])

    distances = jnp.sum((data_point[None, ...] - centroids) ** 2, axis=-1)
    distances = jax.vmap(weight_distance, in_axes=(None, 0, 0, 0))(
        assignment_point, jnp.arange(centroids.shape[0]), cluster_populations, distances
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
    centroids, old_assignments, _, loss, counter = carry

    assignments, centroids = inner_loop_hartigan(old_assignments.copy(), centroids, data)
    # loss = loss.at[counter].set(compute_loss(data, centroids, assignments))
    loss = compute_loss(data, centroids, assignments)
    return (centroids, assignments, old_assignments, loss, counter + 1)


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
    loss = 0.0  # jnp.zeros(max_iters)
    counter = 0

    init_assignments = assign_clusters(init_centroids, data)
    init_centroids = update_centroids(data, init_assignments, init_centroids.shape[0])

    cond_fun = jax.jit(partial(_hart_kmeans_stop_condition, max_steps=max_iters))

    # makes sure the initial assignment does not trigger the stop condition
    carry = (init_centroids, init_assignments, init_assignments - 1, loss, counter)

    @jax.jit
    def run_batched_hartigan_inner(carry):
        return jax.lax.while_loop(
            cond_fun=cond_fun,
            body_fun=lambda c: _hartigan_kmeans_step(c, data),
            init_val=carry,
        )

    centroids, assignments, _, loss, counter = run_batched_hartigan_inner(carry)
    return centroids, assignments, loss, counter


############################ Batched Hartigan ############################
def weight_distances(assignments, cluster_ids, cluster_populations, distances):
    return jax.vmap(
        jax.vmap(weight_distance, in_axes=(None, 0, 0, 0)), in_axes=(0, None, None, 0)
    )(assignments, cluster_ids, cluster_populations, distances)


def assign_dp_to_cluster_batched_hartigan(
    centroids, assignments, cluster_populations, data
):
    distances = jnp.sum((data[:, None, :] - centroids[None, :, :]) ** 2, axis=-1)
    distances = weight_distances(
        assignments, jnp.arange(centroids.shape[0]), cluster_populations, distances
    )
    return jnp.argmin(distances, axis=-1)


def _batched_hartigan_step(carry, data):
    centroids, old_assignments, old_old_assignments, _, loss, counter = carry

    cluster_populations = jnp.bincount(old_assignments, length=centroids.shape[0])
    assignments = assign_dp_to_cluster_batched_hartigan(
        centroids, old_assignments, cluster_populations, data
    )
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


def _batched_hartigan_stop_condition(carry, max_steps):
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


def run_batched_hartigan_kmeans(
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

    cond_fun = jax.jit(partial(_batched_hartigan_stop_condition, max_steps=max_iters))

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
    def run_batched_hartigan_inner(carry):
        return jax.lax.while_loop(
            cond_fun=cond_fun,
            body_fun=lambda c: _batched_hartigan_step(c, data),
            init_val=carry,
        )

    centroids, assignments, _, _, loss, counter = run_batched_hartigan_inner(carry)
    return centroids, assignments, loss, counter


########################## Mini-batch Hartigan ##########################
def _minibatch_hartigan_step(carry, data, batch_size):
    centroids, old_assignments, _, loss, counter, key = carry

    key, subkey = jax.random.split(key)

    cluster_populations = jnp.bincount(old_assignments, length=centroids.shape[0])
    subset_idx = jax.random.choice(
        subkey, data.shape[0], shape=(batch_size,), replace=False
    )
    assignments = assign_dp_to_cluster_batched_hartigan(
        centroids, old_assignments[subset_idx], cluster_populations, data[subset_idx]
    )
    assignments = old_assignments.at[subset_idx].set(assignments)

    centroids = update_centroids(data, assignments, centroids.shape[0])

    loss = compute_loss(data, centroids, assignments)
    return (centroids, assignments, old_assignments, loss, counter + 1, key)


def _minibatch_hartigan_stop_condition(carry, max_steps):
    (
        _,
        assignments,
        old_assignments,
        _,
        counter,
        _,
    ) = carry

    cond1 = jnp.any(assignments != old_assignments)
    cond3 = counter <= max_steps

    return cond1 & cond3  # & cond4


def run_minibatch_hartigan_kmeans(
    data: Float[Array, "n d"],
    init_centroids: Float[Array, "K d"],
    key: PRNGKeyArray,
    batch_size: Int,
    max_iters: Int = 1000,
) -> Tuple[
    Tuple[Float[Array, "K d"], Int[Array, " n"]],
    Float[Array, " max_steps"],
]:
    loss = 0.0
    counter = 0

    init_assignments = assign_clusters(init_centroids, data)
    init_centroids = update_centroids(data, init_assignments, init_centroids.shape[0])
    cond_fun = jax.jit(partial(_minibatch_hartigan_stop_condition, max_steps=max_iters))

    # makes sure the initial assignment does not trigger the stop condition
    carry = (
        init_centroids,
        init_assignments,
        init_assignments - 1,
        loss,
        counter,
        key,
    )

    @jax.jit
    def run_minibatch_hartigan_inner(carry):
        return jax.lax.while_loop(
            cond_fun=cond_fun,
            body_fun=lambda c: _minibatch_hartigan_step(c, data, batch_size),
            init_val=carry,
        )

    centroids, assignments, _, loss, counter, _ = run_minibatch_hartigan_inner(carry)
    return centroids, assignments, loss, counter
