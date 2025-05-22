import os
from functools import partial

import jax
import numpy as np
from tqdm import tqdm


jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp


def _compute_upper_bound(d, prior_variance, noise_variance):
    rho = (1 + 2 * prior_variance / noise_variance) / (
        1 + prior_variance / noise_variance
    ) ** 2
    return rho ** (d / 4)


@partial(jax.jit, static_argnums=3)
def _run_thm_warump_experiment(key, prior_std, noise_std, d, m=0.0):
    key1, key2, keyn = jax.random.split(key, 3)
    mu1 = jax.random.normal(key1, shape=(d,)) * prior_std
    mu2 = jax.random.normal(key2, shape=(d,)) * prior_std
    noise = jax.random.normal(keyn, shape=(d,)) * noise_std
    x = mu1 + noise

    return jnp.sum((x - mu2) ** 2) - jnp.sum((x - mu1) ** 2) <= m


def run_experiments_theorem_warmup(
    dimension_vals,
    noise_std_vals,
    prior_std,
    n_experiments,
    path_to_output,
    *,
    overwrite=False,
    seed=0,
    batch_size=1000,
):
    if os.path.exists(path_to_output) and not overwrite:
        raise ValueError(
            f"Ouput file {path_to_output} exists, but overwrite was set to False"
        )
    else:
        os.makedirs(os.path.dirname(path_to_output), exist_ok=True)

    X, Y = np.meshgrid(dimension_vals, noise_std_vals**2, indexing="ij")
    upper_bound = _compute_upper_bound(d=X, prior_variance=1, noise_variance=Y)
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
