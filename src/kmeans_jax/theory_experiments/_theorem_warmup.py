import os
from functools import partial
from typing import Dict

import jax
import numpy as np
from jaxtyping import Array, Bool, Float, Int, PRNGKeyArray
from tqdm import tqdm


jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp


def _compute_upper_bound(
    dimension: Int, prior_variance: Float, noise_variance: Float
) -> Float:
    rho = (1 + 2 * prior_variance / noise_variance) / (
        1 + prior_variance / noise_variance
    ) ** 2
    return rho ** (dimension / 4)


@partial(jax.jit, static_argnums=3)
def _run_thm_warump_experiment(
    key: PRNGKeyArray, prior_std: Float, noise_std: Float, dimension: Int, m: Int = 0.0
) -> Int:
    key1, key2, keyn = jax.random.split(key, 3)
    mu1 = jax.random.normal(key1, shape=(dimension,)) * prior_std
    mu2 = jax.random.normal(key2, shape=(dimension,)) * prior_std
    noise = jax.random.normal(keyn, shape=(dimension,)) * noise_std
    x = mu1 + noise
    if jnp.sum((x - mu2) ** 2) - jnp.sum((x - mu1) ** 2) <= m:
        return 1
    else:
        return 0


def run_experiments_theorem_warmup(
    dimension_vals: Int[Array, " n_dim_vals"],
    noise_std_vals: Float[Array, " n_noise_std_vals"],
    prior_std: Float,
    n_experiments: Int,
    path_to_output: str,
    *,
    overwrite: Bool = False,
    seed: Int = 0,
    batch_size: Int = 1000,
) -> Dict[str, Array]:
    """
    Run the warmup experiments for Theorem E.1

    **Arguments:**
        dimension_vals: The dimensions to test.
        noise_std_vals: The noise standard deviations to test.
        prior_std: The prior standard deviation.
        n_experiments: The number of experiments to run.
        path_to_output: The path to save the results.
        overwrite: Whether to overwrite the output file if it exists.
        seed: The random seed to use.
        batch_size: The batch size to use for JAX.

    **Returns:**
        A dictionary with the results of the experiments.

    The results are also saved in a .npz file in the specified path.
    """

    if os.path.exists(path_to_output) and not overwrite:
        raise ValueError(
            f"Ouput file {path_to_output} exists, but overwrite was set to False"
        )
    else:
        os.makedirs(os.path.dirname(path_to_output), exist_ok=True)

    X, Y = np.meshgrid(dimension_vals, noise_std_vals**2, indexing="ij")
    upper_bound = _compute_upper_bound(dimension=X, prior_variance=1, noise_variance=Y)
    key = jax.random.key(seed)
    empirical_probs = np.ones((*upper_bound.shape, n_experiments)) * -1.0

    results_warmup = {}

    for i in tqdm(range(len(dimension_vals))):
        for j in range(len(noise_std_vals)):
            key, *subkeys = jax.random.split(key, n_experiments + 1)
            subkeys = jnp.array(subkeys)

            empirical_probs[i, j] = jax.lax.map(
                lambda x: _run_thm_warump_experiment(
                    x, prior_std, noise_std_vals[j], dimension_vals[i]
                ),
                xs=subkeys,
                batch_size=batch_size,
            )
            jax.clear_caches()

            results_warmup = {
                "upper_bound": upper_bound,
                "empirical_probs": empirical_probs,
                "dimension_vals": dimension_vals,
                "noise_std_vals": noise_std_vals,
                "seed": seed,
                "n_experiments": n_experiments,
                "key": jax.random.key_data(key),
            }

            jnp.savez(path_to_output, **results_warmup)

    return results_warmup
