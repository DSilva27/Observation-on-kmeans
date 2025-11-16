from functools import partial
from typing import Optional, Tuple
from typing_extensions import Literal

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp
from jaxtyping import Array, Bool, Float, Int

from kmeans_jax.kmeans import kmeans_plusplus_init, kmeans_random_init


class ExpMax(eqx.Module):
    n_clusters: int
    max_iter: int
    n_init: int = 1
    init: Literal["random", "minibatch", "kmeans++"]
    init_batch_size: int | None = None
    rtol: float = 1e-5
    atol: float = 1e-5

    def __init__(
        self,
        n_clusters: int,
        *,
        n_init: int,
        max_iter: int,
        init: Literal["random", "minibatch", "kmeans++"] = "random",
        init_batch_size: int | None = None,
        rtol: float = 1e-5,
        atol: float = 1e-5,
    ):
        """
        KMeans clustering class inspired by the scikit-learn API.

        **Arguments:**
        - n_clusters: The number of clusters to form.
        - n_init: Number of times the algorithm will be run with different random seeds.
        - max_iter: Maximum number of iterations for a single run.
        - init: Method for initialization.
            - "random": selects random data points as centroids.
            - "minibatch": runs EM using a random minibatch of the data
                and uses the output for initialization.
        - init_batch_size: The size of the minibatch to use for initialization
            when `init` is set to "minibatch".
        """
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.n_init = n_init
        self.init = init

        if init == "random" or init == "kmeans++":
            self.init_batch_size = None

        elif init == "minibatch":
            assert (
                init_batch_size is not None
            ), "init_batch_size must be provided for minibatch init"
            self.init_batch_size = init_batch_size
        else:
            raise ValueError(f"Unknown init method: {init}")

        self.atol = atol
        self.rtol = rtol

    def fit(
        self, key, data, var_noise, var_prior, *, batch_size=None, output="best"
    ) -> dict:
        """
        Fit the KMeans model to the data.

        **Arguments:**
        - key: JAX random key for reproducibility.
        - data: The data to cluster, shape (n, d).
        - var_noise: The noise variance for each data point, shape (n,).
        - var_prior: The prior variance for the centroids (acts as regularization).
        - batch_size: Optional batch size for running the algorithm
            for each initialization in parallel. If None, the algorithm will run
            sequentially for each initialization.

        **Returns:**
        - A dictionary containing the best centroids, responsibilities, and likelihood
        from the `best` initialization, or all results if `output` is set to "all".

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
        assert output in ["best", "all"], f"Unknown output type: {output}"

        results = jax.lax.map(
            lambda x: _run_em_for_wrapper(
                x,
                data,
                var_noise,
                var_prior,
                self.init,
                self.init_batch_size,
                self.n_clusters,
                self.max_iter,
                self.atol,
                self.rtol,
            ),
            keys,
            batch_size=batch_size,
        )

        results["centroids"].block_until_ready()

        if output == "best":
            best_idx = jnp.argmax(results["likelihood"])
            results = {
                "centroids": results["centroids"][best_idx],
                "responsibilities": results["responsibilities"][best_idx],
                "likelihood": results["likelihood"][best_idx],
                "n_iter": results["n_iter"][best_idx],
            }
        else:
            pass

        return results


def _run_em_for_wrapper(
    key,
    data,
    var_noise,
    var_prior,
    init_mode,
    init_batch_size,
    n_clusters,
    max_iter,
    atol,
    rtol,
):
    if init_mode == "random":
        init_centroids, _ = kmeans_random_init(data, n_clusters, key)
    elif init_mode == "kmeans++":
        init_centroids, _ = kmeans_plusplus_init(data, n_clusters, key)

    else:
        key1, key2 = jax.random.split(key)
        batch_idx = jax.random.choice(
            key1, data.shape[0], (init_batch_size,), replace=False
        )
        data_batch = data[batch_idx]
        var_noise_batch = var_noise[batch_idx]

        init_centroids, _, _, _ = run_exp_max(
            data_batch,
            var_noise_batch,
            kmeans_random_init(data, n_clusters, key2)[0],
            var_prior,
            100,
            1e-5,
            1e-5,
        )

    centroids, responsibilities, losses, counter = run_exp_max(
        data, var_noise, init_centroids, var_prior, max_iter, atol, rtol
    )
    return {
        "centroids": centroids,
        "responsibilities": responsibilities,
        "likelihood": losses,
        "n_iter": counter,
    }


############### EM Functions #################


def _compute_log_likelihood_datapoint(
    centroid: Float[Array, " d"],
    data_point: Float[Array, " d"],
    var_noise: Float,
) -> Float:
    return -0.5 * jnp.sum(jnp.abs(data_point - centroid) ** 2, axis=-1) / var_noise


def compute_log_likelihood(
    centroids: Float[Array, "K d"],
    data: Float[Array, "N d"],
    var_noise: Float[Array, " N"],
) -> Float:
    log_likelihoods = _compute_log_likelihood_datapoint(
        centroids[None, :, :], data[:, None, :], var_noise[:, None]
    )

    log_likelihood = jnp.mean(logsumexp(log_likelihoods, axis=-1))

    return log_likelihood


def compute_responsibilities(
    centroids: Float[Array, "K d"],
    data: Float[Array, "N d"],
    var_noise: Float[Array, " N"],
) -> Float[Array, "N K"]:
    # jax.debug.print("centroids shape: {cs}", cs=centroids.shape)
    # jax.debug.print("data shape: {ds}", ds=data.shape)
    # jax.debug.print("var_noise shape: {vns}", vns=var_noise)

    log_likelihoods = _compute_log_likelihood_datapoint(
        centroids[None, :, :], data[:, None, :], var_noise[:, None]
    )
    responsibilities = jnp.exp(
        log_likelihoods - logsumexp(log_likelihoods, axis=1, keepdims=True)
    )
    return responsibilities


def compute_em_centroids(
    data: Float[Array, "N d"],
    var_noise: Float[Array, " N"],
    var_prior: Float,
    responsibilities: Float[Array, "N K"],
) -> Float[Array, "K d"]:
    centroids = responsibilities.T @ (data / var_noise[:, None])
    centroids /= (responsibilities.T @ (1 / var_noise) + 1 / var_prior)[:, None]

    return centroids


def _take_em_step(
    carry: Tuple[Float[Array, "K d"], Float, Float, Int],
    data: Float[Array, "N d"],
    var_noise: Float[Array, " N"],
    var_prior: float,
):
    centroids, old_likelihood, _, counter = carry

    responsibilities = compute_responsibilities(centroids, data, var_noise)
    centroids = compute_em_centroids(data, var_noise, var_prior, responsibilities)
    likelihood = compute_log_likelihood(centroids, data, var_noise)

    # jax.debug.print("likelihood at step {c}: {l}", c=counter, l=likelihood)

    return (centroids, likelihood, old_likelihood, counter + 1)


def _em_stop_condition(
    carry: Tuple[Float[Array, "K d"], Float, Float, Int],
    max_steps: Int,
    atol,
    rtol,
) -> Bool:
    _, likelihood, old_likelihood, counter = carry

    cond1 = jnp.invert(
        jnp.isclose(
            likelihood,
            old_likelihood,
            atol=atol,
            rtol=rtol,
        )
    )
    # jax.debug.print(
    #     "EM step {c}: likelihood = {l}, old_likelihood = {ol}, close = {cl}",
    #     c=counter,
    #     l=likelihood,
    #     ol=old_likelihood,
    #     cl=cond1,
    # )
    cond2 = counter <= max_steps

    # return cond2
    return cond1 & cond2


def run_exp_max(
    data: Float[Array, "n d"],
    var_noise: Float[Array, " n"],
    init_centroids: Float[Array, "K d"],
    var_prior: Optional[Float] = None,
    max_iters: Int = 1000,
    atol: Float = 1e-5,
    rtol: Float = 1e-5,
) -> Tuple[Float[Array, "K d"], Float[Array, " n K"], Float, Int]:
    counter = 0
    if var_prior is None:
        var_prior = 1e8

    cond_fun = jax.jit(
        partial(_em_stop_condition, max_steps=max_iters, atol=atol, rtol=rtol)
    )

    # makes sure the initial assignment does not trigger the stop condition
    carry = (init_centroids, 0.0, -jnp.inf, counter)

    @jax.jit
    def run_em_inner(carry):
        return jax.lax.while_loop(
            cond_fun=cond_fun,
            body_fun=lambda c: _take_em_step(c, data, var_noise, var_prior),
            init_val=carry,
        )

    centroids, likelihood, _, counter = run_em_inner(carry)
    responsibilities = compute_responsibilities(centroids, data, var_noise)
    # likelihood = likelihood[:counter]
    # likelihood = jax.lax.slice(likelihood, (0,), (counter,))

    # centroids.block_until_ready()
    return centroids, responsibilities, likelihood, counter
