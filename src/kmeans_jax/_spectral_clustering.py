import numpy as np
from sklearn.cluster import SpectralClustering

from kmeans_jax.kmeans import compute_centroids, compute_loss


def run_spectral_clustering(data, n_clusters, **kwargs):
    """
    Runs spectral clustering using sklearn's implementation.

    **Arguments:**
        data: np.ndarray of shape (n_samples, n_features)
            The input data to cluster.
        n_clusters: int
            The number of clusters to form.
        **kwargs: additional keyword arguments to pass to sklearn's SpectralClustering.

    **Returns:**
        centroids: np.ndarray of shape (n_clusters, n_features)
            The computed cluster centroids.
        labels: np.ndarray of shape (n_samples,)
            The labels assigned to each data point.
        loss: float
            The k-means loss.
        num_iters: None
            SpectralClustering does not provide iteration count.
    """
    data_norm = data / np.linalg.norm(data, axis=1, keepdims=True)
    clustering = SpectralClustering(n_clusters=n_clusters, **kwargs).fit(data_norm)
    labels = clustering.labels_
    centroids = compute_centroids(data, labels, n_clusters)
    loss = compute_loss(data, centroids, labels)
    num_iters = None  # SpectralClustering does not provide iteration count

    return centroids, labels, loss, num_iters
