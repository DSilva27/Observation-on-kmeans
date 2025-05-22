from functools import partial

import jax


jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp


############### initializations ################
@partial(jax.jit, static_argnums=(1,))
def kmeans_plusplus_init(data, K, key):
    key, subkey = jax.random.split(key)
    init_centroids = jnp.zeros((K, data.shape[-1]), dtype=data.dtype)
    indices = jnp.zeros((K,), dtype=int)

    first_index = jax.random.choice(subkey, data.shape[0])
    indices = indices.at[0].set(first_index)
    init_centroids = init_centroids.at[0].set(data[indices[0]])

    def body_fun(i, val):
        centroids, indices, key = val
        mask = jnp.arange(K) < i
        valid_centroids = jnp.where(mask[:, None], centroids, jnp.inf)

        distances = jnp.linalg.norm(data[:, None, :] - valid_centroids, axis=-1)
        min_distances = jnp.min(distances, axis=1) ** 2

        key, subkey = jax.random.split(key)
        new_index = jax.random.choice(
            subkey, data.shape[0], p=min_distances / jnp.sum(min_distances)
        )
        indices = indices.at[i].set(new_index)
        centroids = centroids.at[i].set(data[new_index])
        return (centroids, indices, key)

    centroids, indices, _ = jax.lax.fori_loop(
        1, K, body_fun, (init_centroids, indices, key)
    )
    return centroids, indices


def kmeans_random_init(data, K, key):
    indices = jax.random.choice(key, data.shape[0], (K,), replace=False)
    return data[indices], indices


############### Lloyd k-means ################
@jax.jit
def compute_loss(data, centroids, cluster_assignments):
    return jnp.sum(jnp.abs(data - centroids[cluster_assignments]) ** 2)


@jax.jit
def assign_clusters(centroids, data):
    distances = jnp.linalg.norm(data[:, None, :] - centroids[None, :, :], axis=-1)
    return jnp.argmin(distances, axis=1)


@partial(jax.jit, static_argnums=(2,))
def update_centroids(data, cluster_assignments, num_clusters):
    def compute_centroid(cluster):
        num = jnp.sum(
            data,
            axis=0,
            where=jnp.where(cluster_assignments == cluster, True, False)[:, None],
        )
        den = jnp.sum(jnp.where(cluster_assignments == cluster, True, False)) + 1e-16
        return num / den

    return jax.vmap(compute_centroid)(jnp.arange(num_clusters))


@jax.jit
def _kmeans_step(carry, data):
    centroids, old_cluster_assignments, _, losses, counter = carry

    cluster_assignments = assign_clusters(centroids, data)
    centroids = update_centroids(data, cluster_assignments, centroids.shape[0])
    losses = losses.at[counter].set(compute_loss(data, centroids, cluster_assignments))
    return (centroids, cluster_assignments, old_cluster_assignments, losses, counter + 1)

def _kmeans_stop_condition(carry, max_steps):
    _, cluster_assignments, old_cluster_assignments, losses, counter = carry

    cond1 = jnp.any(cluster_assignments != old_cluster_assignments)
    cond2 = counter <= max_steps
    
    return cond1 & cond2


def run_kmeans(data, init_centroids, max_iters=1000):

    losses = jnp.zeros(max_iters)
    counter = 0
    # dummy init
    init_assignments = jnp.ones(data.shape[0], dtype=int) * -1

    cond_fun = jax.jit(partial(_kmeans_stop_condition, max_steps=max_iters))
    
    # makes sure the initial assignment does not trigger the stop condition
    carry = (init_centroids, init_assignments, init_assignments-1, losses, counter)

    @jax.jit
    def run_kmeans_inner(carry):
        return jax.lax.while_loop(
            cond_fun=cond_fun, body_fun=lambda c: _kmeans_step(c, data), init_val=carry
        )

    centroids, assigments, _, losses, counter = run_kmeans_inner(carry)
    losses = losses[:counter]

    centroids.block_until_ready()
    return (centroids, assigments), losses
