from typing import TYPE_CHECKING, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import inv_boxcox

if TYPE_CHECKING:
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
) -> "matplotlib.figure.Figure":
    """Plot analyte vs continuous covariate(s) with fitted polynomial ribbon.

    Produces one subplot per continuous covariate. Each subplot shows a scatter
    of the raw analyte values and a filled band representing the polynomial
    fit mu(x) ± n_sigma * sigma(x) back-transformed to the original space.
    When more than one continuous covariate is present, the remaining
    covariate(s) are fixed at their sample median for the ribbon calculation.

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
        Label for the analyte (y-axis). Defaults to ``f"col {target_col}"``.
    n_sigma : float, default=1.0
        Half-width of the ribbon in sigma units.
    n_grid : int, default=300
        Number of evenly spaced points for the smooth prediction grid.
    figsize : tuple of (float, float), optional
        Size of a single subplot. The total figure width is
        ``figsize[0] * n_subplots``. Defaults to ``(6, 5)``.

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

    X = np.asarray(X, dtype=float)

    if target_col is None:
        target_col = normalizer._resolved_target_cols[0]

    fitter = normalizer._fitters[target_col]
    lambda_ = fitter.lambda_

    cont_cols = list(normalizer.continuous_cols)
    n_cont = len(cont_cols)

    if figsize is None:
        figsize = (6, 5)

    fig, axes = plt.subplots(
        1, n_cont, figsize=(figsize[0] * n_cont, figsize[1]), squeeze=False
    )
    axes = axes[0]

    y_raw = X[:, target_col]

    cont_medians = np.array([np.median(X[:, c]) for c in cont_cols])

    for k, cont_col in enumerate(cont_cols):
        ax = axes[k]

        ax.scatter(X[:, cont_col], y_raw, s=5, alpha=0.3, color="steelblue", linewidths=0)

        x_min, x_max = X[:, cont_col].min(), X[:, cont_col].max()
        x_grid = np.linspace(x_min, x_max, n_grid)

        X_cont_grid = np.tile(cont_medians, (n_grid, 1))
        X_cont_grid[:, k] = x_grid

        mu_grid, sigma_grid = fitter.predict_mu_sigma(X_cont_grid)
        sigma_grid = np.maximum(sigma_grid, 1e-6)

        y_center = inv_boxcox(mu_grid, lambda_)

        bc_upper = mu_grid + n_sigma * sigma_grid
        bc_lower = mu_grid - n_sigma * sigma_grid

        if lambda_ > 0:
            bc_lower = np.maximum(bc_lower, -1.0 / lambda_ + 1e-8)
        elif lambda_ < 0:
            bc_lower = np.minimum(bc_lower, -1.0 / lambda_ - 1e-8)

        y_upper = inv_boxcox(bc_upper, lambda_)
        y_lower = inv_boxcox(bc_lower, lambda_)

        ax.fill_between(
            x_grid, y_lower, y_upper,
            alpha=0.35, color="tomato",
            label=f"μ ± {n_sigma}σ",
        )
        ax.plot(x_grid, y_center, lw=2, color="tomato", label="μ(x)")

        xlabel = (
            covariate_labels[k]
            if covariate_labels is not None and k < len(covariate_labels)
            else f"covariate col {cont_col}"
        )
        ylabel = analyte_label if analyte_label is not None else f"col {target_col}"

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.legend(loc="upper left", fontsize=8)

        if n_cont > 1:
            median_val = cont_medians[1 - k]
            other_label = (
                covariate_labels[1 - k]
                if covariate_labels is not None and (1 - k) < len(covariate_labels)
                else f"col {cont_cols[1 - k]}"
            )
            ax.set_title(f"{other_label} fixed at median ({median_val:.2g})", fontsize=9)

    fig.tight_layout()
    return fig


__all__ = ["plot_covariate_space"]
