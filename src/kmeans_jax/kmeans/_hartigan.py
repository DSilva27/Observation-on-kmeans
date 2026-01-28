from functools import partial
from typing import Tuple

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float, Int, PRNGKeyArray
from numba import jit

from ._common_functions import assign_clusters, compute_centroids, compute_loss


@jit
def _compute_loss_np(data, centroids, labels):
    return np.sum(np.abs(data - centroids[labels]) ** 2)


@jit
def _assign_labels_lloyd_np(centroids, data):
    distances = np.sum((data[:, None, :] - centroids[None, :, :]) ** 2, axis=-1)
    return np.argmin(distances, axis=1)


@jit
def _compute_centroids_np(data, labels, centroids):
    for i in range(centroids.shape[0]):
        idx_mask = labels == i
        n_elements = np.sum(idx_mask)
        if n_elements > 0:
            centroids[i] = np.sum(data[idx_mask], axis=0) / n_elements
        else:
            centroids[i] = np.zeros_like(centroids[i])
    return centroids


@jit
def _assign_label_hartigan_np(centroids, cluster_populations, data_point, label_point):
    distances = np.sum((data_point[None, ...] - centroids) ** 2, axis=-1)

    for i in range(centroids.shape[0]):
        if label_point == i:
            if cluster_populations[i] <= 1:
                distances[i] = -1.0  # always assign
            else:
                scale_factor = cluster_populations[i] / (cluster_populations[i] - 1)
                distances[i] *= scale_factor

        else:
            scale_factor = cluster_populations[i] / (cluster_populations[i] + 1)
            distances[i] *= scale_factor
    return np.argmin(distances)


@jit
def _run_hartigan_numpy(data, init_centroids, max_iters):
    max_iters = 100

    # Initial quantities
    labels = _assign_labels_lloyd_np(init_centroids, data)
    centroids = _compute_centroids_np(data, labels, init_centroids.copy())
    cluster_populations = np.bincount(labels, minlength=init_centroids.shape[0])

    # Variables to update
    old_labels = labels.copy()
    n_iters = 0
    for n_iters in range(max_iters):
        for j in range(data.shape[0]):
            new_label = _assign_label_hartigan_np(
                centroids, cluster_populations, data[j], labels[j]
            )
            if new_label != labels[j]:
                # centroids = compute_centroids(data, labels, centroids)
                n_clust1 = cluster_populations[labels[j]]
                centroids[labels[j]] = (centroids[labels[j]] * n_clust1 - data[j]) / (
                    n_clust1 - 1.0
                )

                n_clust2 = cluster_populations[new_label]
                centroids[new_label] = (centroids[new_label] * n_clust2 + data[j]) / (
                    n_clust2 + 1.0
                )

                cluster_populations[labels[j]] -= 1
                cluster_populations[new_label] += 1
                labels[j] = new_label

        if np.array_equal(labels, old_labels):
            break
        else:
            old_labels = labels.copy()

    loss = _compute_loss_np(data, centroids, labels)
    return centroids, labels, loss, n_iters


def run_hartigan_kmeans(
    data: Float[Array, "n d"],
    init_centroids: Float[Array, "K d"],
    max_iters: Int = 1000,
) -> Tuple[Float[Array, "K d"], Int[Array, " n"], Float, Int]:
    """
    Run k-means clustering using online Hartigan's algorithm. Unlike other algorithms
    in the library, this is implemented to run on Numba, and thus has no GPU support.

    **Arguments**:
        data: A numpy-like array of shape (n, d) containing the data points.
        init_centroids: A numpy-like array of shape (K, d)
                        containing the initial centroids.
        max_iters: maximum number of iterations.

    **Returns**:
        A tuple containing:
            - centroids: A numpy-like array of shape (K, d) containing the final centroids
            - labels: A numpy-like array of shape (n,) containing the final cluster labels
            - loss: A float representing the final k-means loss.
            - n_iters: An integer representing the number of iterations performed.
    """
    return _run_hartigan_numpy(
        np.asanyarray(data), np.asanyarray(init_centroids), max_iters
    )


############################ Batched Hartigan ############################
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


def weight_distances(labels, cluster_ids, cluster_populations, distances):
    return jax.vmap(
        jax.vmap(weight_distance, in_axes=(None, 0, 0, 0)), in_axes=(0, None, None, 0)
    )(labels, cluster_ids, cluster_populations, distances)


def assign_dp_to_cluster_batched_hartigan(centroids, labels, cluster_populations, data):
    distances = jnp.sum((data[:, None, :] - centroids[None, :, :]) ** 2, axis=-1)
    distances = weight_distances(
        labels, jnp.arange(centroids.shape[0]), cluster_populations, distances
    )
    return jnp.argmin(distances, axis=-1)


def _batched_hartigan_step(carry, data):
    centroids, old_labels, old_old_labels, _, loss, counter = carry

    cluster_populations = jnp.bincount(old_labels, length=centroids.shape[0])
    labels = assign_dp_to_cluster_batched_hartigan(
        centroids, old_labels, cluster_populations, data
    )
    # jax.debug.print(
    #     "Step {c}, Assignments {a}, cluster populations {p}",
    #     c=counter,
    #     a=old_labels,
    #     p=cluster_populations,
    # )
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


def _batched_hartigan_stop_condition(carry, max_steps):
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


def run_batched_hartigan_kmeans(
    data: Float[Array, "n d"],
    init_centroids: Float[Array, "K d"],
    max_iters: Int = 1000,
) -> Tuple[Float[Array, "K d"], Int[Array, " n"], Float, Int]:
    loss = 0.0  # jnp.zeros(max_iters)
    counter = 0

    init_labels = assign_clusters(init_centroids, data)
    init_centroids = compute_centroids(data, init_labels, init_centroids.shape[0])
    # init_centroids, init_labels, _, _ =
    # run_lloyd_kmeans(data, init_centroids, max_iters=5)

    cond_fun = jax.jit(partial(_batched_hartigan_stop_condition, max_steps=max_iters))

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
    def run_batched_hartigan_inner(carry):
        return jax.lax.while_loop(
            cond_fun=cond_fun,
            body_fun=lambda c: _batched_hartigan_step(c, data),
            init_val=carry,
        )

    centroids, labels, _, _, loss, counter = run_batched_hartigan_inner(carry)
    return centroids, labels, loss, counter


########################## Mini-batch Hartigan ##########################
def _minibatch_hartigan_step(carry, data, batch_size):
    centroids, old_labels, _, loss, counter, key = carry

    key, subkey = jax.random.split(key)

    cluster_populations = jnp.bincount(old_labels, length=centroids.shape[0])
    subset_idx = jax.random.choice(
        subkey, data.shape[0], shape=(batch_size,), replace=False
    )
    labels = assign_dp_to_cluster_batched_hartigan(
        centroids, old_labels[subset_idx], cluster_populations, data[subset_idx]
    )
    labels = old_labels.at[subset_idx].set(labels)

    centroids = compute_centroids(data, labels, centroids.shape[0])

    loss = compute_loss(data, centroids, labels)
    return (centroids, labels, old_labels, loss, counter + 1, key)


def _minibatch_hartigan_stop_condition(carry, max_steps):
    (
        _,
        labels,
        old_labels,
        _,
        counter,
        _,
    ) = carry

    cond1 = jnp.any(labels != old_labels)
    cond3 = counter <= max_steps

    return cond1 & cond3  # & cond4


def run_minibatch_hartigan_kmeans(
    data: Float[Array, "n d"],
    init_centroids: Float[Array, "K d"],
    key: PRNGKeyArray,
    batch_size: Int,
    max_iters: Int = 1000,
) -> Tuple[Float[Array, "K d"], Int[Array, " n"], Float, Int]:
    loss = 0.0
    counter = 0

    init_labels = assign_clusters(init_centroids, data)
    init_centroids = compute_centroids(data, init_labels, init_centroids.shape[0])
    cond_fun = jax.jit(partial(_minibatch_hartigan_stop_condition, max_steps=max_iters))

    # makes sure the initial assignment does not trigger the stop condition
    carry = (
        init_centroids,
        init_labels,
        init_labels - 1,
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

    centroids, labels, _, loss, counter, _ = run_minibatch_hartigan_inner(carry)
    return centroids, labels, loss, counter
