import cvxpy as cp
import numpy as np

from kmeans_jax.kmeans import compute_centroids, compute_loss, assign_clusters
import warnings

warnings.filterwarnings("ignore")


def sdp_rounding_vectorized(denoised, n_clusters):
    """
    Numpy version of the SDP rounding procedure implemented in:
    https://github.com/solevillar/kmeans_sdp/
    """
    d = denoised.shape[1]

    # Vectorized affinity matrix computation
    # Compute pairwise distances between all points
    distances = np.linalg.norm(denoised[:, None] - denoised[None, :], axis=-1)
    affinity = np.where(distances < 1e-3, 1.0, 0.0)

    centers = np.zeros((n_clusters, d))

    for t in range(n_clusters):
        # Find most popular point
        popularity = np.sum(affinity, axis=0)
        idx = np.argmax(popularity)

        # Store center
        centers[t, :] = denoised[idx]

        # Find all points connected to this center
        connected_points = affinity[:, idx] == 1

        # Zero out rows and columns for connected points
        affinity[connected_points, :] = 0
        affinity[:, connected_points] = 0

    # Vectorized labels computation
    # Compute distances from all points to all centers
    labels = assign_clusters(centers, denoised)

    return labels


def run_sdp_clustering(data, n_clusters, max_iters=500):
    n = data.shape[0]
    data = np.asanyarray(data).copy()
    data_normalized = data / np.linalg.norm(data, axis=1, keepdims=True)

    dist_matrix = np.sum(
        (data_normalized[:, None, :] - data_normalized[None, :, :]) ** 2, axis=-1
    )

    n = data.shape[0]
    X = cp.Variable((n, n), symmetric=True)

    constraints = [
        X >> 0,
        cp.trace(X) == n_clusters,
        X >= 0,
        X @ np.ones((n, n)) == np.ones((n, n)),
    ]

    prob = cp.Problem(cp.Minimize(cp.trace(dist_matrix @ X)), constraints)

    prob.solve(solver=cp.SCS, verbose=False, max_iters=max_iters)

    labels = sdp_rounding_vectorized(X.value @ data_normalized, n_clusters)
    centroids = compute_centroids(data, labels, n_clusters)
    loss = compute_loss(data, centroids, labels)
    num_iters = prob.solver_stats.num_iters

    return centroids, labels, loss, num_iters
