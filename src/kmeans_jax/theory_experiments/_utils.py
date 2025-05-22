import numpy as np
from jaxtyping import Array, Float, Int
from scipy import stats


def compute_conf_interval(trials: Int[Array, "... n_trials"], alpha: Float = 0.05):
    """
    Compute the confidence interval for Binomial proportions using
    Wilson's Interval.

    **Arguments:**
        trials: The trials to compute the confidence interval for.
            Shape (..., n_trials) where ... is any number of batch dimensions and
            n is the number oftrials performed.
        alpha: The significance level.
    **Returns:**
        A tuple containing the lower and upper bounds of the confidence interval for
        each batch dimension.
    """
    n_s = trials.sum(axis=-1)
    n = trials.shape[-1]

    crit = stats.norm.isf(alpha)
    crit2 = crit**2
    q = n_s / n
    denom = 1 + crit2 / n
    center = (q + crit2 / (2 * n)) / denom
    width = crit * np.sqrt(q * (1.0 - 1) / n + crit2 / (4.0 * n**2))

    width /= denom

    return center - width, center + width


def compute_conf_interval_with_mask(
    trials: Int[Array, "B1 B2 n_trials"],
    mask: Int[Array, "B1 B2 n_trials"],
    alpha: Float = 0.05,
):
    """
    Compute the confidence interval for Binomial proportions using
    Wilson's Interval. The mask specifies which elements to include in the
    computation.

    **Arguments:**
        trials: The trials to compute the confidence interval for.
            Shape (B1, B2, n_trials) where B1 and B2 are batch dimensions
            and n is the number of trials performed.
        mask: The mask to apply to the trials. Shape (B1, B2, n_trials).
        alpha: The significance level.
    **Returns:**
        A tuple containing the lower and upper bounds of the confidence interval.
    """
    crit = stats.norm.isf(alpha)
    crit2 = crit**2

    def _compute_single_case(p, m):
        n_s = (p * m).sum()
        n = m.sum()

        q = n_s / n
        denom = 1 + crit2 / n
        c = (q + crit2 / (2 * n)) / denom
        w = crit * np.sqrt(q * (1.0 - 1) / n + crit2 / (4.0 * n**2))

        w /= denom

        return c, w

    center, width = (
        np.zeros((trials.shape[0], trials.shape[1])),
        np.zeros((trials.shape[0], trials.shape[1])),
    )

    for i in range(trials.shape[0]):
        for j in range(trials.shape[1]):
            center[i, j], width[i, j] = _compute_single_case(trials[i, j], mask[i, j])

    return center - width, center + width
