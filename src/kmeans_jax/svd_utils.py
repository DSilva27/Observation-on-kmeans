"""
This module provides a JAX implementation of randomized SVD.

The implementation is based on the randomized SVD implementation
from the `sklearn` library.

The algorithm itself comes from the paper:
Halko, N., Martinsson, P. G., & Tropp, J. A. (2011).
Finding structure with randomness: Probabilistic algorithms for constructing approximate
matrix decompositions. SIAM review, 53(2), 217-288.

https://epubs.siam.org/doi/10.1137/090771806
"""

from typing import Tuple
from typing_extensions import Literal

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int, PRNGKeyArray


def randomized_range_finder(
    key: PRNGKeyArray, a: Float[Array, "n_samples n_features"], *, size: Int, n_iter: Int
) -> Float[Array, " n_samples size"]:
    def qr_normalizer(b):
        return jnp.linalg.qr(b, mode="reduced")

    def body_fun(i, Q):
        Q, _ = qr_normalizer(a @ Q)
        Q, _ = qr_normalizer(a.T @ Q)
        return Q

    Q = jax.random.normal(key, shape=(a.shape[1], size))
    Q = jax.lax.fori_loop(0, n_iter, body_fun, init_val=Q)
    Q, _ = qr_normalizer(a @ Q)
    return Q


@eqx.filter_jit
def randomized_svd(
    key: PRNGKeyArray,
    a: Float[Array, "n_samples n_features"],
    n_components: Int,
    *,
    n_oversamples: Int = 10,
) -> Tuple[
    Float[Array, "n_samples n_components"],
    Float[Array, " n_components"],
    Float[Array, "n_components n_features"],
]:
    """
    Computes a randomized SVD of the input matrix `a`.
    This function is based on the randomized SVD implementation from `sklearn',
    and the paper by Halko et al. (2011).

    **Arguments:**
        key: JAX PRNG key for random number generation.
        a: Input matrix of shape (n_samples, n_features).
        n_components: Number of singular values and vectors to compute.
        n_oversamples: Number of oversampling vectors to use (default is 10).

    **Returns:**
        A tuple containing:
        - U: Left singular vectors of shape (n_samples, n_components).
        - S: Singular values of shape (n_components,).
        - Vt: Right singular vectors of shape (n_components, n_features).
    """
    n_random = n_components + n_oversamples
    n_samples, n_features = a.shape

    # n_iter = 7 if n_components < 0.1 * jnp.minimum(n_samples, n_features) else 4

    n_iter = jax.lax.cond(
        n_components < 0.1 * jnp.minimum(n_samples, n_features),
        lambda x: 7,
        lambda x: 4,
        None,
    )

    is_transpose = n_samples < n_features
    a = a.T if is_transpose else a

    Q = randomized_range_finder(key, a, size=n_random, n_iter=n_iter)

    B = Q.T @ a
    Uhat, S, Vt = jnp.linalg.svd(B, full_matrices=False)
    U = Q @ Uhat

    if is_transpose:
        return Vt[:n_components].T, S[:n_components], U[:, :n_components].T
    else:
        return U[:, :n_components], S[:n_components], Vt[:n_components, :].T


def principal_component_analysis(
    key: PRNGKeyArray,
    data: Float[Array, "n_samples n_features"],
    n_components: Int,
    mode=Literal["randomized", "full"],
    *,
    n_oversamples: Int = 10,
):
    data -= jnp.mean(data, axis=0, keepdims=True)

    if mode == "full":
        U, S, _ = jnp.linalg.svd(data, full_matrices=False)
        U = U[:, :n_components]
        S = S[:n_components]

    elif mode == "randomized":
        U, S, _ = randomized_svd(
            key, data, n_components=n_components, n_oversamples=n_oversamples
        )
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'randomized' or 'full'.")

    return U @ jnp.diag(S)
