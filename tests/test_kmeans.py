import jax
import jax.numpy as jnp
import numpy as np
import pytest

from kmeans_jax.kmeans import kmeans_random_init, run_kmeans


def _update_assignments(data, centroids):
    distances = jnp.linalg.norm(data[:, None] - centroids[None, :], axis=-1)
    return jnp.argmin(distances, axis=1)


def _update_centroids(data, assignments, centroids):
    for i in range(centroids.shape[0]):
        centroids[i] = jnp.mean(data[assignments == i], axis=0)
    return centroids


def lloyd_kmeans_numpy(data, init_centroids, max_iters=1000):
    data = np.asarray(data)
    init_centroids = np.asarray(init_centroids)

    centroids = init_centroids.copy()
    old_assignments = jnp.zeros(data.shape[0], dtype=jnp.int32)
    for counter in range(max_iters):
        assignments = _update_assignments(data, centroids)
        centroids = _update_centroids(data, assignments, centroids)

        if jnp.all(assignments == old_assignments):
            break
        old_assignments = assignments
    return centroids, assignments, counter


def generate_data(key, n_clusters, dimension, cluster_sizes, noise_variance):
    var_prior = 1.0
    key_centers, key_noise = jax.random.split(key, 2)

    # Generate data
    true_centers = jax.random.normal(
        key_centers, shape=(n_clusters, dimension)
    ) * jnp.sqrt(var_prior)
    true_labels = jnp.arange(n_clusters).repeat(cluster_sizes)

    data = true_centers[true_labels] + jax.random.normal(
        key_noise, shape=(true_labels.shape[0], dimension)
    ) * jnp.sqrt(noise_variance)

    return data, true_centers, true_labels


@pytest.mark.parametrize("dimension", [10, 50, 100])
@pytest.mark.parametrize("noise_variance", [1.0, 4.0, 9.0])
def test_lloyd_kmeans(dimension, noise_variance):
    key = jax.random.key(0)

    key_data, key_init = jax.random.split(key, 2)

    n_clusters = 5
    size_per_cluster = 100
    size_clusters = jnp.ones(n_clusters, dtype=jnp.int32) * size_per_cluster

    data, _, _ = generate_data(
        key_data, n_clusters, dimension, size_clusters, noise_variance
    )

    init_centroids, _ = kmeans_random_init(data, n_clusters, key=key_init)

    centroids_np, assignments_np, counter_np = lloyd_kmeans_numpy(
        data, init_centroids, max_iters=1000
    )

    centroids_jax, assignments_jax, _, counter_jax = run_kmeans(
        data, init_centroids, max_iters=1000
    )

    assert jnp.allclose(centroids_np, centroids_jax, atol=1e-5), "Centroids do not match"
    assert (assignments_np == assignments_jax).all(), "Assignments do not match"
    assert counter_np == counter_jax, "Counter does not match"
