import numpy as np
from scipy import stats


def compute_conf_interval(probabilities, alpha=0.05):
    n_s = probabilities.sum(axis=-1)
    n = probabilities.shape[-1]

    crit = stats.norm.isf(alpha)
    crit2 = crit**2
    q = n_s / n
    denom = 1 + crit2 / n
    center = (q + crit2 / (2 * n)) / denom
    width = crit * np.sqrt(q * (1.0 - 1) / n + crit2 / (4.0 * n**2))

    width /= denom

    return center - width, center + width


def compute_conf_interval_with_mask(probabilities, mask, alpha=0.05):
    crit = stats.norm.isf(alpha)
    crit2 = crit**2

    def compute_single_case(p, m):
        n_s = (p * m).sum()
        n = m.sum()

        q = n_s / n
        denom = 1 + crit2 / n
        c = (q + crit2 / (2 * n)) / denom
        w = crit * np.sqrt(q * (1.0 - 1) / n + crit2 / (4.0 * n**2))

        w /= denom

        return c, w

    center, width = (
        np.zeros((probabilities.shape[0], probabilities.shape[1])),
        np.zeros((probabilities.shape[0], probabilities.shape[1])),
    )

    for i in range(probabilities.shape[0]):
        for j in range(probabilities.shape[1]):
            center[i, j], width[i, j] = compute_single_case(
                probabilities[i, j], mask[i, j]
            )

    return center - width, center + width
