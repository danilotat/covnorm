from typing import TYPE_CHECKING, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import inv_boxcox

if TYPE_CHECKING:
    import matplotlib.axes
    import matplotlib.figure

    from covnorm._normalizer import RobustConditionalNormalizer


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

    ax.add_collection3d(Line3DCollection(segments, colors=seg_colors, linewidths=0.6, alpha=.6))

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


__all__ = ["plot_covariate_space"]
