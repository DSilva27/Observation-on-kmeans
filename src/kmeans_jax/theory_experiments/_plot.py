import jax.numpy as jnp
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from ._main_theorem import _compute_rho
from ._typical_partition_theorem import check_partition_is_valid
from ._utils import compute_conf_interval, compute_conf_interval_with_mask


def plot_theorem_warmup(results, fig_fname=None):
    dimension_vals = results["dimension_vals"]
    empirical_probs = results["empirical_probs"]
    upper_bound = results["upper_bound"]
    noise_std_vals = results["noise_std_vals"]
    n_experiments = results["n_experiments"]

    ci_lower, ci_upper = compute_conf_interval(empirical_probs)
    avg_probs = empirical_probs.mean(axis=-1)
    sns.set_theme(context="talk")
    fig, ax = plt.subplots(2, 4, figsize=(15, 10), sharex=False, sharey=True)

    mpl.rcParams["pdf.fonttype"] = 42  # TrueType fonts
    mpl.rcParams["ps.fonttype"] = 42

    for i in range(len(noise_std_vals)):
        if i == len(noise_std_vals) - 1:
            label1 = "Upper Bound"
            label2 = "Empirical Probability"
        else:
            label1 = None
            label2 = None

        sns.lineplot(
            x=dimension_vals,
            y=upper_bound[:, i],
            ls=":",
            marker="o",
            label=label1,
            ax=ax.flatten()[i],
        )

        sns.lineplot(
            x=dimension_vals,
            y=avg_probs[:, i],
            ls="--",
            marker="X",
            label=label2,
            ax=ax.flatten()[i],
        )
        ax.flatten()[i].fill_between(
            dimension_vals, y1=ci_lower[:, i], y2=ci_upper[:, i], alpha=0.3
        )

        ax.flatten()[i].set_xlabel("dimension [d]")
        ax.flatten()[i].set_xticks([10**1, 10**2, 10**3, 10**4, 10**5])

    for i in range(len(noise_std_vals)):
        ax.flatten()[i].set_title(r"$\sigma^2 = $" + f"{noise_std_vals[i] ** 2:.1f}")
        ax.flatten()[i].set_xscale("log")
        ax.flatten()[i].set_yscale("log")
        ax.flatten()[i].set_ylim(1.0 / n_experiments, 1.5)

    ax[0, 0].set_ylabel("")
    ax[1, 0].set_ylabel("")

    ax.flatten()[-2].legend(loc=(1.2, 0.70))
    fig.subplots_adjust(
        left=0.08, right=0.95, top=0.9, bottom=0.1, wspace=0.3, hspace=0.4
    )

    ax.flatten()[-1].set_axis_off()

    for i in range(len(noise_std_vals)):
        ax.flatten()[i].set_xticks([10**1, 10**2, 10**3, 10**4, 10**5])

    if fig_fname is not None:
        fig.savefig(fig_fname, dpi=300, facecolor="white", bbox_inches="tight")

    return


def plot_theorem_diff(results, fig_fname=None):
    empirical_probs_worst = results["empirical_probs_worst"]
    empirical_probs_random = results["empirical_probs_random"]

    upper_bounds_theorem = results["upper_bound"]

    dimension_vals = results["dimension_vals"]
    noise_std_vals = results["noise_std_vals"]
    n_experiments = results["n_experiments"]

    ci_worst = compute_conf_interval(empirical_probs_worst)
    ci_random = compute_conf_interval(empirical_probs_random)
    avg_probs_worst = empirical_probs_worst.mean(axis=-1)
    avg_probs_random = empirical_probs_random.mean(axis=-1)
    sns.set_theme(context="talk")
    mpl.rcParams["pdf.fonttype"] = 42  # TrueType fonts
    mpl.rcParams["ps.fonttype"] = 42

    fig, ax = plt.subplots(
        2, 4, figsize=(15, 10), sharex=False, sharey=True
    )  # , layout="compressed")

    for i in range(len(noise_std_vals)):
        if i == len(noise_std_vals) - 1:
            label1 = "Upper Bound"
            label2 = "Empirical Probability - Worst"
            label3 = "Empirical Probability - Random"
        else:
            label1 = None
            label2 = None
            label3 = None

        sns.lineplot(
            x=dimension_vals,
            y=upper_bounds_theorem[:, i],
            ls=":",
            marker="o",
            label=label1,
            ax=ax.flatten()[i],
        )

        sns.lineplot(
            x=dimension_vals,
            y=avg_probs_worst[:, i],
            ls="--",
            marker="X",
            label=label2,
            ax=ax.flatten()[i],
        )
        ax.flatten()[i].fill_between(
            dimension_vals, y1=ci_worst[0][:, i], y2=ci_worst[1][:, i], alpha=0.3
        )

        sns.lineplot(
            x=dimension_vals,
            y=avg_probs_random[:, i],
            ls="-.",
            marker="^",
            label=label3,
            ax=ax.flatten()[i],
        )
        ax.flatten()[i].fill_between(
            dimension_vals, y1=ci_random[0][:, i], y2=ci_random[1][:, i], alpha=0.3
        )

        ax.flatten()[i].set_xlabel("dimension [d]")
        ax.flatten()[i].set_xticks([10**1, 10**2, 10**3, 10**4, 10**5])

    for i in range(len(noise_std_vals)):
        ax.flatten()[i].set_title(r"$\sigma^2 = $" + f"{noise_std_vals[i] ** 2:.0f}")
        ax.flatten()[i].set_xscale("log")
        ax.flatten()[i].set_yscale("log")
        ax.flatten()[i].set_ylim(1.0 / n_experiments, 1.5)

    for i in range(len(noise_std_vals)):
        ax.flatten()[i].set_xticks([10**1, 10**2, 10**3, 10**4, 10**5])

    ax[0, 0].set_ylabel("")
    ax[1, 0].set_ylabel("")

    ax.flatten()[-2].legend(loc=(1.2, 0.70))
    fig.subplots_adjust(
        left=0.08, right=0.95, top=0.9, bottom=0.1, wspace=0.3, hspace=0.4
    )

    ax.flatten()[-1].set_axis_off()

    if fig_fname is not None:
        fig.savefig(fig_fname, dpi=300, facecolor="white", bbox_inches="tight")
    return


def plot_theorem_typical_partition(results, fig_fname=None):
    # Results from experiments
    empirical_probs = results["empirical_probs"]
    upper_bound_theorem = results["upper_bound"]
    cluster_sizes = results["cluster_sizes"]

    # parameters used to run them
    q_value = results["q_value"]
    n_data_points = results["n_data_points"]
    dimension_vals = results["dimension_vals"]
    beta_vals = results["beta_vals"]
    n_experiments = results["n_experiments"]

    mask = check_partition_is_valid(
        cluster_sizes[..., 0], cluster_sizes[..., 1], n_data_points, q_value
    )

    ci_lower, ci_upper = compute_conf_interval_with_mask(empirical_probs, mask)
    center = empirical_probs.mean(axis=-1, where=jnp.where(mask, True, False))

    sns.set_theme(context="talk")
    mpl.rcParams["pdf.fonttype"] = 42  # TrueType fonts
    mpl.rcParams["ps.fonttype"] = 42

    fig, ax = plt.subplots(2, 4, figsize=(15, 10), sharex=False, sharey=True)

    for i in range(empirical_probs.shape[1]):
        if i == empirical_probs.shape[1] - 1:
            label1 = "Upper Bound"
            label2 = "Empirical Probability"
        else:
            label1 = None
            label2 = None

        sns.lineplot(
            x=dimension_vals,
            y=upper_bound_theorem[:, i],
            ls=":",
            marker="o",
            label=label1,
            ax=ax.flatten()[i],
        )

        sns.lineplot(
            x=dimension_vals,
            y=center[:, i],
            ls="--",
            marker="X",
            label=label2,
            ax=ax.flatten()[i],
        )
        ax.flatten()[i].fill_between(
            dimension_vals, y2=ci_lower[:, i], y1=ci_upper[:, i], alpha=0.3
        )

        ax.flatten()[i].set_xlabel("dimension [d]")
        ax.flatten()[i].set_xticks([10**1, 10**2, 10**3, 10**4, 10**5])

    for i in range(empirical_probs.shape[1]):
        ax.flatten()[i].set_title(r"$\beta = $" + f"{beta_vals[i]}")
        ax.flatten()[i].set_xscale("log")
        ax.flatten()[i].set_yscale("log")
        ax.flatten()[i].set_ylim(1.0 / n_experiments, 1.5)
        ax.flatten()[i].set_xticks([10**1, 10**2, 10**3, 10**4, 10**5])

    ax[0, 0].set_ylabel("")
    ax[1, 0].set_ylabel("")

    ax.flatten()[-2].legend(loc=(1.2, 0.70))
    fig.subplots_adjust(
        left=0.08, right=0.95, top=0.9, bottom=0.1, wspace=0.3, hspace=0.4
    )

    ax.flatten()[-1].set_axis_off()

    if fig_fname is not None:
        fig.savefig(fig_fname, dpi=300, facecolor="white", bbox_inches="tight")

    return


def plot_union_bound_corollary(results, fig_fname=None):
    # Results from experiments
    empirical_probs = 1.0 - (results["empirical_probs"] == 0.0)
    upper_bound = results["upper_bound"]

    # parameters used to run them
    dimension_vals = results["dimension_vals"]
    noise_std_vals = results["noise_std_vals"]
    n_experiments = results["n_experiments"]

    ci_cor = compute_conf_interval(empirical_probs)
    avg_probs_cor = empirical_probs.mean(axis=-1)

    sns.set_theme(context="talk")
    mpl.rcParams["pdf.fonttype"] = 42  # TrueType fonts
    mpl.rcParams["ps.fonttype"] = 42

    fig, ax = plt.subplots(2, 4, figsize=(15, 10), sharex=False, sharey=True)

    for i in range(len(noise_std_vals)):
        if i == len(noise_std_vals) - 1:
            label1 = "Upper Bound"
            label2 = "Empirical Probability"
        else:
            label1 = None
            label2 = None

        sns.lineplot(
            x=dimension_vals,
            y=upper_bound[:, i],
            ls=":",
            marker="o",
            label=label1,
            ax=ax.flatten()[i],
        )

        sns.lineplot(
            x=dimension_vals,
            y=avg_probs_cor[:, i],
            ls="--",
            marker="X",
            label=label2,
            ax=ax.flatten()[i],
        )
        ax.flatten()[i].fill_between(
            dimension_vals, y1=ci_cor[0][:, i], y2=ci_cor[1][:, i], alpha=0.3
        )

        ax.flatten()[i].set_xlabel("dimension [d]")
        ax.flatten()[i].set_xticks([10**1, 10**2, 10**3, 10**4, 10**5])

    for i in range(len(noise_std_vals)):
        ax.flatten()[i].set_title(r"$\sigma^2 = $" + f"{noise_std_vals[i] ** 2:.1f}")
        ax.flatten()[i].set_xscale("log")
        ax.flatten()[i].set_yscale("log")
        ax.flatten()[i].set_ylim(1.0 / n_experiments, 1.5)

    ax[0, 0].set_ylabel("")
    ax[1, 0].set_ylabel("")

    ax.flatten()[-2].legend(loc=(1.2, 0.70))
    fig.subplots_adjust(
        left=0.08, right=0.95, top=0.9, bottom=0.1, wspace=0.3, hspace=0.4
    )

    ax.flatten()[-1].set_axis_off()

    for i in range(len(noise_std_vals)):
        ax.flatten()[i].set_xticks([10**1, 10**2, 10**3, 10**4, 10**5])

    if fig_fname is not None:
        fig.savefig(fig_fname, dpi=300, facecolor="white", bbox_inches="tight")
    return


###### Values of rho ######


def _assumption_noise_variance_lb(
    prior_variance,
    size_cluster_C,
    size_cluster_T,
):
    return (
        2
        * prior_variance
        * (size_cluster_C - 1) ** 2
        / (size_cluster_C**2 / size_cluster_T + size_cluster_C)
    )


def _compute_rho_equal_clusters(
    s_param,
    noise_variance,
):
    prior_variance = 1.0
    X, Y = np.meshgrid(s_param, noise_variance, indexing="ij")
    rho = _compute_rho(
        prior_variance=prior_variance,
        noise_variance=Y,
        size_cluster_C=X,
        size_cluster_T=X,
    )

    rho_satisf_assumption = (
        _assumption_noise_variance_lb(
            prior_variance=prior_variance,
            size_cluster_C=X,
            size_cluster_T=X,
        )
        < Y
    )

    noise_var_lb_1d = _assumption_noise_variance_lb(
        prior_variance=prior_variance,
        size_cluster_C=s_param,
        size_cluster_T=s_param,
    )

    grid = {
        "s_values": X,
        "noise_variance": Y,
    }
    return rho, rho_satisf_assumption, noise_var_lb_1d, grid


def _compute_rho_second_cluster_half(
    s_param,
    noise_variance,
):
    prior_variance = 1.0
    X, Y = np.meshgrid(s_param, noise_variance, indexing="ij")
    rho = _compute_rho(
        prior_variance=prior_variance,
        noise_variance=Y,
        size_cluster_C=X,
        size_cluster_T=X / 2,
    )

    rho_mask = (
        _assumption_noise_variance_lb(
            prior_variance=prior_variance,
            size_cluster_C=X,
            size_cluster_T=X / 2,
        )
        < Y
    )

    noise_var_lb_1d = _assumption_noise_variance_lb(
        prior_variance=prior_variance,
        size_cluster_C=s_param,
        size_cluster_T=s_param / 2,
    )

    grid = {
        "s_values": X,
        "noise_variance": Y,
    }
    return rho, rho_mask, noise_var_lb_1d, grid


def _compute_rho_second_cluster_large(
    s_param,
    noise_variance,
):
    prior_variance = 1.0
    X, Y = np.meshgrid(s_param, noise_variance, indexing="ij")
    rho = _compute_rho(
        prior_variance=prior_variance,
        noise_variance=Y,
        size_cluster_C=X,
        size_cluster_T=10e6,
    )

    rho_mask = (
        _assumption_noise_variance_lb(
            prior_variance=prior_variance,
            size_cluster_C=X,
            size_cluster_T=10e6,
        )
        < Y
    )

    noise_var_lb_1d = _assumption_noise_variance_lb(
        prior_variance=prior_variance,
        size_cluster_C=s_param,
        size_cluster_T=10e6,
    )

    grid = {
        "s_values": X,
        "noise_variance": Y,
    }
    return rho, rho_mask, noise_var_lb_1d, grid


def _compute_rho_small_prior(
    s_param,
    noise_variance,
):
    prior_variance = 1e-8
    X, Y = np.meshgrid(s_param, noise_variance, indexing="ij")
    rho = _compute_rho(
        prior_variance=prior_variance,
        noise_variance=Y,
        size_cluster_C=X,
        size_cluster_T=X,
    )

    rho_mask = (
        _assumption_noise_variance_lb(
            prior_variance=prior_variance,
            size_cluster_C=X,
            size_cluster_T=X,
        )
        < Y
    )

    noise_var_lb_1d = _assumption_noise_variance_lb(
        prior_variance=prior_variance,
        size_cluster_C=s_param,
        size_cluster_T=s_param,
    )

    grid = {
        "s_values": X,
        "noise_variance": Y,
    }
    return rho, rho_mask, noise_var_lb_1d, grid


def _plot_rho(
    ax, s, noise_variance, compute_rho_fn, fs=14, xlabel=None, ylabel=None, title=None
):
    rho_values, rho_mask, noise_var_lb, grid = compute_rho_fn(
        s_param=s,
        noise_variance=noise_variance,
    )

    rho_masked = np.where(rho_mask, rho_values, np.nan)

    # dashed lines
    ax.plot(noise_variance, noise_variance, "r--")  # s = sigma^2

    # valid region
    ax.plot(s, noise_var_lb, "b--")

    # edge of the region
    ax.plot(s, np.zeros(100), ls="--", color="blue")
    ax.plot(np.ones(100) * np.max(s), noise_var_lb, ls="--", color="blue")

    mask_for_upper_line = noise_var_lb >= np.max(noise_variance)
    if np.any(mask_for_upper_line):
        ax.plot(
            s[mask_for_upper_line],
            np.ones_like(s[mask_for_upper_line]) * np.max(noise_variance),
            ls="--",
            color="blue",
        )

    levels = [0.965, 0.98, 0.99, 0.995]
    levels1 = np.arange(0.965, 1.0, 0.005)

    contour = ax.contourf(
        grid["s_values"],
        grid["noise_variance"],
        rho_masked,
        cmap="YlOrBr_r",
        levels=levels1,
    )
    cbar = plt.colorbar(contour)
    cbar.set_ticks(levels1[::2].tolist())
    cbar.set_ticklabels(levels1[::2].tolist())
    cbar.set_label(r"$\rho$ (Theorem 2.6)", fontsize=fs)

    ax.contour(
        grid["s_values"],
        grid["noise_variance"],
        rho_masked,
        colors="black",
        levels=levels,
    )

    ax.set_xlim((-0.9, 51.0))
    ax.set_ylim((-0.9, 51.0))

    # Labels and title
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=fs)
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=fs)
    if title is not None:
        ax.set_title(title, fontsize=fs)

    return


def plot_rho_paper_figure(fig_fname=None):
    mpl.rcParams["pdf.fonttype"] = 42  # TrueType fonts
    mpl.rcParams["ps.fonttype"] = 42

    fig, ax = plt.subplots(
        2, 2, figsize=(10, 8), layout="compressed", sharex=True, sharey=True
    )
    fs = 14

    noise_variance = np.linspace(1.0, 50.0, 100)
    s = np.linspace(1, 50, 100)

    _plot_rho(
        ax[0, 0],
        s=s,
        noise_variance=noise_variance,
        fs=fs,
        compute_rho_fn=_compute_rho_equal_clusters,
        ylabel=r"$\sigma^2$ [Noise variance]",
        title=r"$s_C = s_T = s$ and $\tau = 1$",
    )

    _plot_rho(
        ax[0, 1],
        s=s,
        noise_variance=noise_variance,
        fs=fs,
        compute_rho_fn=_compute_rho_second_cluster_half,
        title=r"$s_C = s, s_T = s/2$ and $\tau = 1$",
    )

    _plot_rho(
        ax[1, 0],
        s=s,
        noise_variance=noise_variance,
        fs=fs,
        compute_rho_fn=_compute_rho_second_cluster_large,
        ylabel=r"$\sigma^2$ [Noise variance]",
        xlabel=r"$s$ [Cluster size]",
        title=r"$s_C = s, s_T \rightarrow \infty$ and $\tau = 1$",
    )

    _plot_rho(
        ax[1, 1],
        s=s,
        noise_variance=noise_variance,
        fs=fs,
        compute_rho_fn=_compute_rho_small_prior,
        xlabel=r"$s$ [Cluster size]",
        title=r"$s_C = s_T = s$ and $\tau \rightarrow 0$",
    )

    if fig_fname is not None:
        fig.savefig(fig_fname, dpi=300, facecolor="white", bbox_inches="tight")

    plt.show()
    return
