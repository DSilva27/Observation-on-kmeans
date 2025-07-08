from typing import Callable
from typing_extensions import Literal

import equinox as eqx
import jax
import jax.numpy as jnp

from ._hartigan import run_batched_hartigan_kmeans, run_hartigan_kmeans
from ._init_methods import (
    kmeans_init_from_random_partition,
    kmeans_plusplus_init,
    kmeans_random_init,
)
from ._lloyd import (
    run_kmeans,
)


class KMeans(eqx.Module):
    n_clusters: int
    max_iter: int
    init_function: Callable
    clustering_function: Callable
    n_init: int = 1

    def __init__(
        self,
        n_clusters: int,
        *,
        n_init: int,
        max_iter: int,
        init: Literal["random", "kmeans++", "random partition"],
        algorithm: Literal["Hartigan", "Batched Hartigan", "Lloyd"],
    ):
        """
        KMeans clustering class inspired by the scikit-learn API.

        **Arguments:**
        - n_clusters: The number of clusters to form.
        - n_init: Number of times the algorithm will be run with different random seeds.
        - max_iter: Maximum number of iterations for a single run.
        - init: Method for initialization.
            - "random": selects random data points as centroids.
            - "kmeans++": initializes centroids using the k-means++ method.
            - "random partition": initializes centroids by randomly partitioning
               the data into clusters and averaging them.
        - algorithm: The clustering algorithm to use.
            - "Lloyd" uses the classic Lloyd's algorithm.
            - "Hartigan" uses the Hartigan's algorithm.
            - "Batched Hartigan" uses a batched version of Hartigan's algorithm.
        """
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.n_init = n_init

        if init == "random":
            self.init_function = kmeans_random_init
        elif init == "kmeans++":
            self.init_function = kmeans_plusplus_init
        elif init == "random partition":
            self.init_function = kmeans_init_from_random_partition
        else:
            raise ValueError(f"Unknown init method: {init}")

        if algorithm == "Hartigan":
            self.clustering_function = run_hartigan_kmeans
        elif algorithm == "Batched Hartigan":
            self.clustering_function = run_batched_hartigan_kmeans
        elif algorithm == "Lloyd":
            self.clustering_function = run_kmeans
        else:
            raise ValueError(f"Unknown clustering method: {algorithm}")

    def fit(self, key, data, *, batch_size=None, output="best") -> dict:
        """
        Fit the KMeans model to the data.

        **Arguments:**
        - key: JAX random key for reproducibility.
        - data: The data to cluster, shape (n, d).
        - batch_size: Optional batch size for running the algorithm
            for each initialization in parallel. If None, the algorithm will run
            sequentially for each initialization.

        **Returns:**
        - A dictionary containing the best centroids, labels, and loss from the
        `best` initialization, or all results if `output` is set to "all".

        **Example:**
        ```python
        import jax
        from kmeans_jax import KMeans

        key = jax.random.key(0)
        key_data, key_kmeans = jax.random.split(key)

        data = function_that_generates_data(key_data, ...)
        kmeans = KMeans(
            n_clusters=3, n_init=10, max_iter=100, init="kmeans++", algorithm="Lloyd"
        )
        results = kmeans.fit(key_kmeans, data, batch_size=5)
        """
        keys = jax.random.split(key, self.n_init)
        results = jax.lax.map(
            lambda x: _run_kmeans_from_data(
                x,
                data,
                self.init_function,
                self.clustering_function,
                self.n_clusters,
                self.max_iter,
            ),
            keys,
            batch_size=batch_size,
        )

        results["centroids"].block_until_ready()

        if output == "best":
            best_idx = jnp.argmin(results["loss"])
            results = {
                "centroids": results["centroids"][best_idx],
                "labels": results["labels"][best_idx],
                "loss": results["loss"][best_idx],
            }
        elif output == "all":
            pass
        else:
            raise ValueError(f"Unknown output type: {output}")

        return results


@eqx.filter_jit
def _run_kmeans_from_data(key, data, init_fn, clustering_fn, n_clusters, max_iter):
    init_centroids, _ = init_fn(data, n_clusters, key)

    (centroids, labels), losses = clustering_fn(data, init_centroids, max_iter)

    return {
        "centroids": centroids,
        "labels": labels,
        "loss": losses,
    }
