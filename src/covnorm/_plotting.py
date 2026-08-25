from math import ceil
from typing import TYPE_CHECKING, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import ArrayLike
from scipy import stats
from scipy.special import inv_boxcox

from covnorm._surface_fitter import _ZSCORE_OUTLIER_THRESHOLD

if TYPE_CHECKING:
    import matplotlib.axes
    import matplotlib.figure

    from covnorm._normalizer import RobustConditionalNormalizer


def plot_worm(
    normalizer: "RobustConditionalNormalizer",
    X: ArrayLike,
    marker_col: int = 0,
    *,
    categorical_vals: Optional[ArrayLike] = None,
    continuous_vals: Optional[ArrayLike] = None,
    covariate_index: int = 0,
    n_bins: int = 6,
    alpha: float = 0.05,
    covariate_label: Optional[str] = None,
    marker_label: Optional[str] = None,
    figsize: Optional[Tuple[float, float]] = None,
) -> "matplotlib.figure.Figure":
    """Plot conditional detrended Q-Q plots of normalized marker values.

    The plotted values are the final Z-scores returned by ``normalizer.transform``.
    When continuous covariates are present, observations are split into equal-count,
    non-overlapping bins of one selected covariate. A well-calibrated normalizer
    produces worms that fluctuate around zero inside the pointwise normal-reference
    bands.

    Parameters
    ----------
    normalizer : fitted RobustConditionalNormalizer
        Normalizer used to compute the diagnostic Z-scores.
    X : array-like of shape (n_samples, n_markers) or (n_samples,)
        Raw marker values. As with :meth:`RobustConditionalNormalizer.transform`,
        covariates are supplied separately.
    marker_col : int, default=0
        Marker column in ``X`` to diagnose.
    categorical_vals : array-like, optional
        Categorical covariates for ``X``. Defaults to the values stored on the
        normalizer. Pass this for new samples whose rows differ from the training
        data.
    continuous_vals : array-like, optional
        Continuous covariates for ``X``. Defaults to the values stored on the
        normalizer. Pass this for new samples whose rows differ from the training
        data.
    covariate_index : int, default=0
        Continuous-covariate column used to form conditional panels. Ignored when
        the normalizer has no continuous covariates.
    n_bins : int, default=6
        Requested number of equal-count conditional panels. It is reduced when
        necessary to keep at least two observations per panel. With no continuous
        covariates, one unconditional panel is drawn.
    alpha : float, default=0.05
        Pointwise error rate for the normal-reference bands.
    covariate_label : str, optional
        Label for the conditioning covariate. Defaults to
        ``"continuous covariate {covariate_index}"``.
    marker_label : str, optional
        Label used in the figure title. Defaults to ``"marker {marker_col}"``.
    figsize : tuple of (float, float), optional
        Figure size. By default it is derived from the panel layout using
        ``3.6`` by ``2.8`` inches per panel.

    Returns
    -------
    matplotlib.figure.Figure
        Figure containing the worm-plot panels.

    Notes
    -----
    The confidence bands are exact pointwise bands for normal order statistics.
    They do not account for fitting the normalizer on the same observations, so a
    held-out or out-of-fold ``X`` gives the most informative calibration check.

    Each panel reports the median Z-score, the normal-consistent MAD scale, and
    the percentages below and above the normalizer's outlier threshold
    (``-3.372`` and ``+3.372``). A calibrated panel has median and tail percentages
    near zero and MAD scale near one.
    """
    if not normalizer._fitters:
        raise ValueError("Normalizer has not been fitted. Call fit() first.")
    if isinstance(marker_col, bool) or not isinstance(marker_col, (int, np.integer)):
        raise TypeError("marker_col must be an integer.")
    if isinstance(n_bins, bool) or not isinstance(n_bins, (int, np.integer)):
        raise TypeError("n_bins must be an integer.")
    if n_bins < 1:
        raise ValueError("n_bins must be at least 1.")
    if not np.isfinite(alpha) or not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be strictly between 0 and 1.")

    z_scores = normalizer.transform(
        X,
        categorical_vals=categorical_vals,
        continuous_vals=continuous_vals,
    )
    n_samples, n_markers = z_scores.shape
    if marker_col < 0 or marker_col >= n_markers:
        raise IndexError(
            f"marker_col={marker_col} is out of range for X with {n_markers} columns."
        )
    if n_samples < 2:
        raise ValueError("A worm plot requires at least two observations.")

    residuals = np.asarray(z_scores[:, marker_col], dtype=float)
    if not np.all(np.isfinite(residuals)):
        raise ValueError("Normalized marker values must all be finite.")

    cont_source = (
        continuous_vals if continuous_vals is not None else normalizer.continuous_vals
    )
    cont_data = normalizer._coerce_covariates(cont_source, n_samples)
    if cont_data.shape[0] != n_samples:
        raise ValueError(
            f"continuous_vals has {cont_data.shape[0]} rows but X has {n_samples}."
        )

    if cont_data.shape[1] == 0:
        groups = [np.arange(n_samples)]
        titles = [f"All observations (n={n_samples})"]
    else:
        if isinstance(covariate_index, bool) or not isinstance(
            covariate_index, (int, np.integer)
        ):
            raise TypeError("covariate_index must be an integer.")
        if covariate_index < 0 or covariate_index >= cont_data.shape[1]:
            raise IndexError(
                f"covariate_index={covariate_index} is out of range for "
                f"continuous_vals with {cont_data.shape[1]} columns."
            )
        try:
            covariate = np.asarray(cont_data[:, covariate_index], dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError("The conditioning covariate must be numeric.") from exc
        if not np.all(np.isfinite(covariate)):
            raise ValueError(
                "The conditioning covariate must contain only finite values."
            )

        label = (
            covariate_label
            if covariate_label is not None
            else f"continuous covariate {covariate_index}"
        )
        groups = _equal_count_groups(covariate, n_bins)
        titles = [
            _format_bin_title(label, covariate[group], len(group)) for group in groups
        ]

    n_panels = len(groups)
    n_cols = min(3, n_panels)
    n_rows = ceil(n_panels / n_cols)
    if figsize is None:
        figsize = (3.6 * n_cols, 2.8 * n_rows)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=figsize,
        squeeze=False,
        sharex=True,
        sharey=True,
    )
    flat_axes = axes.ravel()

    for panel, (ax, row_indices, title) in enumerate(zip(flat_axes, groups, titles)):
        panel_residuals = residuals[row_indices]
        theoretical, deviation, lower, upper = _worm_coordinates(panel_residuals, alpha)
        ax.fill_between(
            theoretical,
            lower,
            upper,
            color="0.9",
            label=f"{100 * (1 - alpha):g}% pointwise band",
        )
        ax.axhline(0.0, color="0.35", linewidth=1.0, linestyle="--")
        ax.scatter(
            theoretical,
            deviation,
            s=12,
            color="black",
            alpha=0.65,
            linewidths=0,
            zorder=3,
        )

        degree = min(3, len(theoretical) - 1)
        coefficients = np.polyfit(theoretical, deviation, degree)
        line_x = np.linspace(theoretical[0], theoretical[-1], 200)
        ax.plot(
            line_x,
            np.polyval(coefficients, line_x),
            color="tomato",
            linewidth=2.0,
            label="cubic trend" if degree == 3 else "trend",
        )
        ax.set_title(title, fontsize=9)
        ax.text(
            0.02,
            0.98,
            _format_worm_statistics(panel_residuals),
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=7,
            linespacing=1.25,
            bbox={
                "boxstyle": "round,pad=0.25",
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.8,
            },
            zorder=4,
        )
        ax.grid(alpha=0.15)
        if panel == 0:
            ax.legend(loc="best", fontsize=8)

    for ax in flat_axes[n_panels:]:
        fig.delaxes(ax)

    fig_width, fig_height = fig.get_size_inches()
    layout_left = min(0.18, 0.55 / fig_width)
    layout_bottom = min(0.20, 0.42 / fig_height)
    layout_right = 1.0 - min(0.03, 0.10 / fig_width)
    layout_top = max(0.60, 1.0 - 0.42 / fig_height)

    name = marker_label if marker_label is not None else f"marker {marker_col}"
    suptitle = fig.suptitle(f"Worm plot — {name}", y=1.0 - min(0.12, 0.10 / fig_height))
    supxlabel = fig.supxlabel(
        "Theoretical normal quantile", y=min(0.08, 0.11 / fig_height)
    )
    supylabel = fig.supylabel(
        "Observed − theoretical quantile", x=min(0.08, 0.11 / fig_width)
    )
    suptitle.set_in_layout(False)
    supxlabel.set_in_layout(False)
    supylabel.set_in_layout(False)
    fig.tight_layout(rect=(layout_left, layout_bottom, layout_right, layout_top))
    return fig


def _equal_count_groups(covariate: np.ndarray, n_bins: int) -> List[np.ndarray]:
    """Return stable, non-overlapping groups with approximately equal counts."""
    if np.ptp(covariate) == 0.0:
        return [np.arange(len(covariate))]
    effective_bins = min(n_bins, max(1, len(covariate) // 2))
    order = np.argsort(covariate, kind="mergesort")
    return [group for group in np.array_split(order, effective_bins) if len(group) > 0]


def _format_bin_title(label: str, values: np.ndarray, count: int) -> str:
    lower = np.min(values)
    upper = np.max(values)
    return f"{label}: {lower:.4g}–{upper:.4g} (n={count})"


def _worm_coordinates(
    residuals: np.ndarray, alpha: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return detrended Q-Q coordinates and exact pointwise normal bands."""
    observed = np.sort(np.asarray(residuals, dtype=float))
    n = len(observed)
    ranks = np.arange(1, n + 1)
    probabilities = (ranks - 3.0 / 8.0) / (n + 1.0 / 4.0)
    theoretical = stats.norm.ppf(probabilities)
    deviation = observed - theoretical

    lower_probability = stats.beta.ppf(alpha / 2.0, ranks, n + 1 - ranks)
    upper_probability = stats.beta.ppf(1.0 - alpha / 2.0, ranks, n + 1 - ranks)
    lower = stats.norm.ppf(lower_probability) - theoretical
    upper = stats.norm.ppf(upper_probability) - theoretical
    return theoretical, deviation, lower, upper


def _worm_statistics(
    residuals: np.ndarray,
) -> Tuple[float, float, float, float]:
    """Return median, normal-consistent MAD, and lower/upper tail fractions."""
    residuals = np.asarray(residuals, dtype=float)
    median = float(np.median(residuals))
    mad = float(np.median(np.abs(residuals - median)))
    mad_scale = mad / float(stats.norm.ppf(0.75))
    lower_tail = float(np.mean(residuals < -_ZSCORE_OUTLIER_THRESHOLD))
    upper_tail = float(np.mean(residuals > _ZSCORE_OUTLIER_THRESHOLD))
    return median, mad_scale, lower_tail, upper_tail


def _format_worm_statistics(residuals: np.ndarray) -> str:
    median, mad_scale, lower_tail, upper_tail = _worm_statistics(residuals)
    return (
        f"median={median:+.2f}  MADσ={mad_scale:.2f}\n"
        f"tail−={100 * lower_tail:.2f}%  tail+={100 * upper_tail:.2f}%"
    )


def plot_covariate_space(
    normalizer: "RobustConditionalNormalizer",
    X: np.ndarray,
    target_col: Optional[int] = None,
    covariate_labels: Optional[List[str]] = None,
    analyte_label: Optional[str] = None,
    n_sigma: float = 1.0,
    n_grid: int = 300,
    figsize: Optional[Tuple[float, float]] = None,
    ax: Optional["matplotlib.axes.Axes"] = None,
) -> "matplotlib.figure.Figure":
    """Plot analyte vs continuous covariate(s) with fitted polynomial surface.

    For one continuous covariate, produces a scatter of raw analyte values with
    a filled ribbon mu(x) ± n_sigma * sigma(x) back-transformed to the original
    space.  For two continuous covariates, produces a 3-D surface plot of mu(x1,
    x2) with the per-bin estimates scattered on top.

    Parameters
    ----------
    normalizer : fitted RobustConditionalNormalizer
    X : ndarray of shape (n_samples, n_cols)
        The data matrix used for fitting (or a new sample with the same layout).
    target_col : int, optional
        Which target column to visualise. Defaults to the first resolved
        target column (``normalizer._resolved_target_cols[0]``).
    covariate_labels : list of str, optional
        Axis label for each continuous covariate. Length must equal
        ``len(normalizer.continuous_cols)``.
    analyte_label : str, optional
        Label for the analyte axis. Defaults to ``f"col {target_col}"``.
    n_sigma : float, default=1.0
        Half-width of the ribbon in sigma units (1-D case only).
    n_grid : int, default=300
        Number of evenly spaced points per axis for the prediction grid.
    figsize : tuple of (float, float), optional
        Figure size. Defaults to ``(6, 5)`` for 1-D and ``(8, 6)`` for 2-D.
        Ignored when ``ax`` is provided.
    ax : matplotlib.axes.Axes, optional
        Axes to draw into. For the 2-D case a 3-D axes
        (``projection='3d'``) must be passed. When ``None`` a new figure
        and axes are created.

    Returns
    -------
    fig : matplotlib.figure.Figure

    Raises
    ------
    ImportError
        If matplotlib is not installed.
    ValueError
        If the normalizer has not been fitted yet.
    """
    if not normalizer._fitters:
        raise ValueError("Normalizer has not been fitted. Call fit() first.")

    X = normalizer._encode(X)

    if target_col is None:
        target_col = normalizer._resolved_target_cols[0]

    fitter = normalizer._fitters[target_col]
    lambda_ = fitter.lambda_

    cont_cols = list(normalizer.continuous_cols)
    n_cont = len(cont_cols)

    ylabel = analyte_label if analyte_label is not None else f"col {target_col}"

    def _xlabel(k: int) -> str:
        if covariate_labels is not None and k < len(covariate_labels):
            return covariate_labels[k]
        return f"covariate col {cont_cols[k]}"

    if n_cont == 2:
        return _plot_surface_3d(
            fitter,
            X,
            target_col,
            cont_cols,
            lambda_,
            ylabel,
            _xlabel,
            n_grid,
            figsize,
            ax,
        )

    if ax is not None:
        fig = ax.figure
    else:
        if figsize is None:
            figsize = (6, 5)
        fig, ax = plt.subplots(1, 1, figsize=figsize)

    y_raw = X[:, target_col]
    cont_col = cont_cols[0]

    ax.scatter(X[:, cont_col], y_raw, s=2, alpha=0.4, color="black", linewidths=0)

    x_min, x_max = X[:, cont_col].min(), X[:, cont_col].max()
    x_grid = np.linspace(x_min, x_max, n_grid).reshape(-1, 1)

    mu_grid, sigma_grid = fitter.predict_mu_sigma(x_grid)
    sigma_grid = np.maximum(sigma_grid, 1e-6)

    y_center = inv_boxcox(mu_grid, lambda_)

    bc_upper = mu_grid + n_sigma * sigma_grid
    bc_lower = mu_grid - n_sigma * sigma_grid
    if lambda_ > 0:
        bc_lower = np.maximum(bc_lower, -1.0 / lambda_ + 1e-8)
    elif lambda_ < 0:
        bc_lower = np.minimum(bc_lower, -1.0 / lambda_ - 1e-8)

    ax.fill_between(
        x_grid.ravel(),
        inv_boxcox(bc_lower, lambda_),
        inv_boxcox(bc_upper, lambda_),
        alpha=0.35,
        color="tomato",
        label=f"μ ± {n_sigma}σ",
    )
    ax.plot(x_grid.ravel(), y_center, lw=2, color="tomato", label="μ(x)")

    ax.set_xlabel(_xlabel(0))
    ax.set_ylabel(ylabel)
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    return fig


def _plot_surface_3d(
    fitter,
    X: np.ndarray,
    target_col: int,
    cont_cols: List[int],
    lambda_: float,
    zlabel: str,
    xlabel_fn,
    n_grid: int,
    figsize: Optional[Tuple[float, float]],
    ax=None,
) -> "matplotlib.figure.Figure":
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    x0 = np.linspace(X[:, cont_cols[0]].min(), X[:, cont_cols[0]].max(), n_grid)
    x1 = np.linspace(X[:, cont_cols[1]].min(), X[:, cont_cols[1]].max(), n_grid)
    X0, X1 = np.meshgrid(x0, x1)

    X_cont_grid = np.column_stack([X0.ravel(), X1.ravel()])
    mu_flat, _ = fitter.predict_mu_sigma(X_cont_grid)
    mu_surface = inv_boxcox(mu_flat, lambda_).reshape(X0.shape)

    if ax is not None:
        fig = ax.figure
    else:
        if figsize is None:
            figsize = (8, 6)
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection="3d")

    mesh_stride = max(1, n_grid // 40)
    from mpl_toolkits.mplot3d.art3d import Line3DCollection

    norm = plt.Normalize(mu_surface.min(), mu_surface.max())
    cmap = plt.cm.jet

    ii = list(range(0, n_grid, mesh_stride))
    jj = list(range(0, n_grid, mesh_stride))

    segments: list = []
    seg_colors: list = []

    for i in ii:
        for k in range(len(jj) - 1):
            j0, j1 = jj[k], jj[k + 1]
            segments.append(
                [
                    (X0[i, j0], X1[i, j0], mu_surface[i, j0]),
                    (X0[i, j1], X1[i, j1], mu_surface[i, j1]),
                ]
            )
            seg_colors.append(cmap(norm((mu_surface[i, j0] + mu_surface[i, j1]) / 2)))

    for j in jj:
        for k in range(len(ii) - 1):
            i0, i1 = ii[k], ii[k + 1]
            segments.append(
                [
                    (X0[i0, j], X1[i0, j], mu_surface[i0, j]),
                    (X0[i1, j], X1[i1, j], mu_surface[i1, j]),
                ]
            )
            seg_colors.append(cmap(norm((mu_surface[i0, j] + mu_surface[i1, j]) / 2)))

    ax.add_collection3d(
        Line3DCollection(segments, colors=seg_colors, linewidths=0.6, alpha=0.6)
    )

    ax.scatter(
        X[:, cont_cols[0]],
        X[:, cont_cols[1]],
        X[:, target_col],
        s=3,
        color="black",
        alpha=0.3,
        linewidths=0,
    )

    ax.set_xlim(X0.min(), X0.max())
    ax.set_ylim(X1.min(), X1.max())
    ax.set_zlim(
        min(mu_surface.min(), X[:, target_col].min()),
        max(mu_surface.max(), X[:, target_col].max()),
    )

    ax.set_xlabel(xlabel_fn(0))
    ax.set_ylabel(xlabel_fn(1))
    ax.set_zlabel(zlabel)
    fig.tight_layout()
    return fig


__all__ = ["plot_covariate_space", "plot_worm"]
