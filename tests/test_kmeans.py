import jax


jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import pytest

from kmeans_jax.kmeans import kmeans_random_init, run_hartigan_kmeans, run_lloyd_kmeans


def run_lloyd_kmeans_numpy(data, init_centroids, max_iters=1000):
    def _assign_labels(data, centroids):
        distances = jnp.linalg.norm(data[:, None] - centroids[None, :], axis=-1)
        return jnp.argmin(distances, axis=1)

    def _compute_centroids(data, labels, centroids):
        for i in range(centroids.shape[0]):
            centroids[i] = jnp.mean(data[labels == i], axis=0)
        return centroids

    data = np.asarray(data)
    init_centroids = np.asarray(init_centroids)

    centroids = init_centroids.copy()
    old_labels = jnp.zeros(data.shape[0], dtype=jnp.int32)
    for counter in range(max_iters):
        labels = _assign_labels(data, centroids)
        centroids = _compute_centroids(data, labels, centroids)

        if jnp.all(labels == old_labels):
            break
        old_labels = labels
    return centroids, labels, counter


def run_hartigan_kmeans_numpy(data, init_centroids, max_iters=1000):
    def _compute_labels_lloyd(centroids, data):
        distances = np.sum((data[:, None, :] - centroids[None, :, :]) ** 2, axis=-1)
        return np.argmin(distances, axis=1)

    def _compute_centroids(data, labels, centroids):
        for i in range(centroids.shape[0]):
            centroids[i] = np.sum(data[labels == i], axis=0) / np.sum(labels == i)
        return centroids

    def _compute_label_hartigan(centroids, cluster_populations, data_point, label_point):
        distances = np.sum((data_point[None, ...] - centroids) ** 2, axis=-1)

        for i in range(centroids.shape[0]):
            if label_point == i:
                if cluster_populations[i] <= 1:
                    distances[i] = 0.0
                else:
                    scale_factor = cluster_populations[i] / (cluster_populations[i] - 1)
                    distances[i] *= scale_factor

            else:
                scale_factor = cluster_populations[i] / (cluster_populations[i] + 1)
                distances[i] *= scale_factor
        return np.argmin(distances)

    max_iters = 100

    # Initial quantities
    labels = _compute_labels_lloyd(init_centroids, data)
    centroids = _compute_centroids(data, labels, init_centroids.copy())
    cluster_populations = np.bincount(labels, minlength=init_centroids.shape[0])

    # Variables to update
    old_labels = labels.copy()
    n_iters = 0
    for n_iters in range(max_iters):
        for j in range(data.shape[0]):
            new_label = _compute_label_hartigan(
                centroids, cluster_populations, data[j], labels[j]
            )
            if new_label != labels[j]:
                # centroids = _compute_centroids(data, labels, centroids)
                n_clust = cluster_populations[labels[j]]
                centroids[labels[j]] = (centroids[labels[j]] * n_clust - data[j]) / (
                    n_clust - 1.0
                )

                n_clust = cluster_populations[new_label]
                centroids[new_label] = (centroids[new_label] * n_clust + data[j]) / (
                    n_clust + 1.0
                )

                cluster_populations[labels[j]] -= 1
                cluster_populations[new_label] += 1
                labels[j] = new_label

        if np.array_equal(labels, old_labels):
            break
        else:
            old_labels = labels.copy()

    return centroids, labels, n_iters


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

    centroids_np, labels_np, counter_np = run_lloyd_kmeans_numpy(
        data, init_centroids, max_iters=1000
    )

    centroids_jax, labels_jax, _, counter_jax = run_lloyd_kmeans(
        data, init_centroids, max_iters=1000
    )

    assert jnp.allclose(centroids_np, centroids_jax, atol=1e-5), "Centroids do not match"
    assert (labels_np == labels_jax).all(), "Assignments do not match"
    assert counter_np == counter_jax, "Counter does not match"


@pytest.mark.parametrize("dimension", [10, 50, 100])
@pytest.mark.parametrize("noise_variance", [1.0, 4.0, 9.0])
def test_hartigan_kmeans(dimension, noise_variance):
    key = jax.random.key(0)

    key_data, key_init = jax.random.split(key, 2)

    n_clusters = 5
    size_per_cluster = 100
    size_clusters = jnp.ones(n_clusters, dtype=jnp.int32) * size_per_cluster

    data, _, _ = generate_data(
        key_data, n_clusters, dimension, size_clusters, noise_variance
    )

    init_centroids, _ = kmeans_random_init(data, n_clusters, key=key_init)

    centroids_np, labels_np, counter_np = run_hartigan_kmeans_numpy(
        np.asanyarray(data), np.asanyarray(init_centroids), max_iters=1000
    )

    centroids_jax, labels_jax, _, counter_jax = run_hartigan_kmeans(
        data, init_centroids, max_iters=1000
    )

    assert jnp.allclose(centroids_np, centroids_jax, atol=1e-5), "Centroids do not match"
    assert (labels_np == labels_jax).all(), "Assignments do not match"
    assert counter_np == counter_jax, "Counter does not match"
