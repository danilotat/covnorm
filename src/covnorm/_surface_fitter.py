import warnings
from dataclasses import dataclass
from math import comb
from typing import List, Optional, Tuple

import numpy as np
import scipy.stats as stats
from scipy.stats import boxcox, yeojohnson
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KDTree
from sklearn.preprocessing import PolynomialFeatures


@dataclass
class RobustNormalizerConfig:
    """Hard limits shared across normalizer components.

    Attributes
    ----------
    MAX_CATEGORICAL : int
        Maximum number of categorical covariates accepted by
        :class:`RobustConditionalNormalizer`.
    MAX_CONTINUOUS : int
        Maximum number of continuous covariates accepted by
        :class:`RobustConditionalNormalizer`. One covariate uses 1D rolling
        windows; two covariates use k-NN overlapping windows in the scaled
        2D space, which avoids the curse of dimensionality (ADHERENCE Fix 4).
    MIN_BIN_SAMPLES : int
        Minimum number of samples required in a bin to attempt
        Q-Q regression. Bins below this threshold are skipped.
        Mørkved et al. (2015) use ~120 per bin; 30 is the practical
        minimum for a generalised implementation.
    """

    MAX_CATEGORICAL: int = 2
    MAX_CONTINUOUS: int = 2
    MIN_BIN_SAMPLES: int = 30


class ContinuousSurfaceFitter:
    """Polynomial surface model for mu and sigma over one or two continuous covariates.

    For one covariate, sorts the data and creates rolling overlapping windows of
    ``bin_size`` samples, estimates mu and sigma per window via Q-Q regression on
    Box-Cox transformed data, then fits a polynomial curve through the window
    estimates.  For two covariates, k-NN overlapping windows of ``bin_size``
    nearest neighbours in the scaled 2D space are used instead. At prediction time the
    curve is evaluated at the exact covariate values of each sample.

    Outlier removal is iterative and conditional: after each polynomial fit the
    per-sample conditional Z-scores ``(BoxCox(y) - mu(x)) / sigma(x)`` are
    computed and samples with ``|Z| > 3.372`` are dropped before the next fit.

    When no continuous covariates are present, or when the number of valid
    windows is insufficient for stable polynomial fitting, the estimator falls
    back to a single global (mu, sigma) pair derived from the full group.
    All mu and sigma values live in Box-Cox transformed space.

    Parameters
    ----------
    n_bins : int
        Target number of rolling windows. Controls the stride as
        ``stride = (n - bin_size) // (n_bins - 1)``.
    degree : int, default=3
        Degree of the polynomial curve fitted to the window estimates.
        Mørkved et al. (2015) use cubic polynomials.
    lambda_ : float or None, default=None
        Box-Cox transformation (or Jeo-Yohnson) parameter. When ``None`` the
        optimal value is found via a grid search over [-2, 2] that maximises the Pearson correlation between sorted BoxCox(y) values and their theoretical
        normal quantiles (Q-Q linearity). When provided, that value is used
        directly and stored unchanged.
    n_iterations : int, default=3
        Maximum number of iterative conditional outlier-removal passes.
        Each pass refits the polynomial surface and removes samples whose
        conditional Z-score exceeds 3.372 in absolute value.
    log_transform_continuous : bool, default=False
        When ``True``, ``log10`` is applied to all continuous covariate
        columns at the start of :meth:`fit` and :meth:`predict_mu_sigma`.
        Requires all covariate values to be strictly positive.
    bin_size : int, default=120
        Number of samples per rolling window. Mørkved et al. (2015) use 120.
    zero_handles: str, default=`eps`
        Strategy to handles zeros in input data. The default behavior is to use
        Box-Cox transformations, but in case of values equal to 0, the method
        will use any of the strategy here. Could be one of `eps`, `yeojohnson`

    Attributes
    ----------
    poly_transformer : sklearn.preprocessing.PolynomialFeatures
        Transforms covariate vectors into polynomial feature matrices.
    mu_model : sklearn.linear_model.LinearRegression
        Linear model predicting mu (in Box-Cox space) from polynomial features.
    sigma_model : sklearn.linear_model.LinearRegression
        Linear model predicting sigma (in Box-Cox space) from polynomial features.
    lambda_ : float
        Box-Cox (or Jeo-Yohnson) parameter; ``None`` until :meth:`fit` is called.
    _global_mu : float
        Fallback mu (in Box-Cox space) used when the polynomial surface
        cannot be fitted.
    _global_sigma : float
        Fallback sigma (in Box-Cox space) used when the polynomial surface
        cannot be fitted.
    _is_fitted : bool
        ``True`` if the polynomial surface was successfully fitted,
        ``False`` if the fallback values are in use.
    bin_centers_ : ndarray of shape (n_bins, n_features) or None
        Covariate coordinates of each valid bin used in the final fitting
        iteration. ``None`` until :meth:`fit` is called with a surface fit.
    bin_mu_ : ndarray of shape (n_bins,) or None
        Q-Q regression mu estimate for each bin in ``bin_centers_``.
    bin_sigma_ : ndarray of shape (n_bins,) or None
        Q-Q regression sigma estimate for each bin in ``bin_centers_``.

    Notes
    -----
    Q-Q regression estimates mu and sigma by fitting a line through the
    empirical quantiles of Box-Cox (or Jeo-Yohnson) transformed window data
    against the corresponding theoretical normal quantiles using the Blom plotting
    position formula ``(i - 3/8) / (n + 1/4)``. The slope gives sigma
    and the intercept gives mu, both in Box-Cox space. Sigma is floored
    at ``1e-6`` to prevent division by zero during Z-score computation.
    """

    def __init__(
        self,
        n_bins: int,
        degree: int = 3,
        lambda_: Optional[float] = None,
        n_iterations: int = 3,
        log_transform_continuous: bool = False,
        bin_size: int = 120,
        zero_handles: str = "eps",
    ):
        self.n_bins = n_bins
        self.degree = degree
        self.lambda_ = lambda_
        self.n_iterations = n_iterations
        self.log_transform_continuous = log_transform_continuous
        self.bin_size = bin_size
        self.poly_transformer = PolynomialFeatures(
            degree=self.degree, include_bias=True
        )
        self.mu_model = LinearRegression(fit_intercept=False)
        self.sigma_model = LinearRegression(fit_intercept=False)
        self._global_mu: float = 0.0
        self._global_sigma: float = 1.0
        self._is_fitted: bool = False
        self.zero_handles = zero_handles
        self.bin_centers_: Optional[np.ndarray] = None
        self.bin_mu_: Optional[np.ndarray] = None
        self.bin_sigma_: Optional[np.ndarray] = None

    def fit(self, X_cont: np.ndarray, y: np.ndarray) -> "ContinuousSurfaceFitter":
        """Fit the polynomial mu/sigma curve to rolling window estimates.

        Parameters
        ----------
        X_cont : ndarray of shape (n_samples, n_features)
            Continuous covariate matrix. Pass an array with zero columns
            (``n_features == 0``) to trigger the global fallback.
            Accepts one or two columns; two-column input uses k-NN binning.
            All values must be strictly positive when
            ``log_transform_continuous=True``.
        y : ndarray of shape (n_samples,)
            Target values. Must be strictly positive (required by Box-Cox).

        Returns
        -------
        self : ContinuousSurfaceFitter

        Raises
        ------
        ValueError
            If ``log_transform_continuous`` is ``True`` and any continuous
            covariate value is <= 0.
        """
        n_samples, n_features = X_cont.shape

        if self.log_transform_continuous and n_features > 0:
            for c in range(n_features):
                if np.any(X_cont[:, c] <= 0):
                    raise ValueError(
                        f"log_transform_continuous=True requires strictly positive "
                        f"covariate values; column {c} contains values <= 0."
                    )
            X_cont = np.log10(X_cont)

        if self.lambda_ is None:
            self.lambda_ = self._find_lambda_grid_search(y)

        if n_features == 0:
            self._fit_fallback(y)
            return self

        # Iterative conditional outlier removal: fit surface, compute
        # conditional Z-scores, drop |Z| > 3.372, repeat.
        y_work = y.copy()
        X_work = X_cont.copy()

        for _ in range(self.n_iterations):
            if X_work.shape[1] == 1:
                valid_centers, mu_estimates, sigma_estimates = (
                    self._create_rolling_bins(X_work[:, 0], y_work)
                )
            else:
                valid_centers, mu_estimates, sigma_estimates = self._create_knn_bins(
                    X_work, y_work
                )

            n_features = X_work.shape[1]
            min_centers = comb(self.degree + n_features, n_features)
            if len(valid_centers) < min_centers:
                warnings.warn(
                    "Too few valid bins for stable polynomial fitting. "
                    "Falling back to global estimates."
                )
                self._fit_fallback(y_work)
                return self

            X_poly = self.poly_transformer.fit_transform(np.array(valid_centers))
            self.mu_model.fit(X_poly, mu_estimates)
            self.sigma_model.fit(X_poly, sigma_estimates)
            self._is_fitted = True
            self.bin_centers_ = np.array(valid_centers)
            self.bin_mu_ = np.array(mu_estimates)
            self.bin_sigma_ = np.array(sigma_estimates)

            X_poly_work = self.poly_transformer.transform(X_work)
            mu_pred = self.mu_model.predict(X_poly_work)
            sigma_pred = np.maximum(self.sigma_model.predict(X_poly_work), 1e-6)
            y_bc = self._transform(y_work, self.lambda_)

            z = (y_bc - mu_pred) / sigma_pred
            mask = np.abs(z) <= 3.372

            if mask.all():
                break
            if not mask.any():
                warnings.warn(
                    "Outlier rejection would remove all remaining samples "
                    "(the iterative refit degenerated, likely because the "
                    "target is near-constant); keeping this iteration's fit "
                    "instead of continuing."
                )
                break
            y_work = y_work[mask]
            X_work = X_work[mask]

        return self

    def predict_mu_sigma(self, X_cont: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict mu and sigma (in Box-Cox space) for each sample.

        Parameters
        ----------
        X_cont : ndarray of shape (n_samples, n_features)
            Continuous covariate matrix. Must have the same number of
            columns as the array passed to :meth:`fit`. All values must
            be strictly positive when ``log_transform_continuous=True``.

        Returns
        -------
        mu : ndarray of shape (n_samples,)
            Predicted location parameter (in Box-Cox space) for each sample.
        sigma : ndarray of shape (n_samples,)
            Predicted scale parameter (in Box-Cox space). Always positive.
        """
        if not self._is_fitted or X_cont.shape[1] == 0:
            return (
                np.full(X_cont.shape[0], self._global_mu),
                np.full(X_cont.shape[0], self._global_sigma),
            )
        if self.log_transform_continuous:
            X_cont = np.log10(X_cont)
        X_poly = self.poly_transformer.transform(X_cont)
        return self.mu_model.predict(X_poly), self.sigma_model.predict(X_poly)

    def _transform(self, data: np.ndarray, lambda_: float) -> np.ndarray:
        """Apply Box-Cox or Yeo-Johnson transform, honouring zero_handles."""
        if np.any(data <= 0):
            if self.zero_handles.lower() == "eps":
                return boxcox(data + 1e-6, lmbda=lambda_)
            else:
                return yeojohnson(data, lmbda=lambda_)
        return boxcox(data, lmbda=lambda_)

    def _find_lambda_grid_search(self, y: np.ndarray, n_points: int = 41) -> float:
        """Find Box-Cox lambda that maximises Q-Q linearity (Pearson R).

        Evaluates the Pearson correlation between sorted ``BoxCox(y, lam)``
        values and their Blom theoretical normal quantiles across a uniform
        grid of ``n_points`` lambda values in [-2, 2]. The lambda yielding
        the highest correlation is returned.

        Parameters
        ----------
        y : ndarray of shape (n_samples,)
            Strictly positive target values.
        n_points : int, default=41
            Number of lambda candidates to evaluate.

        Returns
        -------
        best_lambda : float
        """
        lambdas = np.linspace(-2.0, 2.0, n_points)
        n = len(y)
        probs = (np.arange(1, n + 1) - 3 / 8) / (n + 1 / 4)
        z_theoretical = stats.norm.ppf(probs)
        best_lambda = 0.0
        best_r = -np.inf
        for lam in lambdas:
            try:
                y_bc = np.sort(self._transform(y, lam))
            except Exception:
                continue
            r = float(np.corrcoef(z_theoretical, y_bc)[0, 1])
            if r > best_r:
                best_r = r
                best_lambda = float(lam)
        return best_lambda

    def _create_rolling_bins(
        self, x: np.ndarray, y: np.ndarray
    ) -> Tuple[List, List, List]:
        """Build rolling overlapping windows sorted by the continuous covariate.

        The stride between consecutive windows is
        ``max(1, (n - bin_size) // (n_bins - 1))``, which targets
        approximately ``n_bins`` windows. Each window's x-coordinate for the
        polynomial fit is the median of the covariate values inside it.

        Parameters
        ----------
        x : ndarray of shape (n_samples,)
            Continuous covariate values (already log-transformed if applicable).
        y : ndarray of shape (n_samples,)
            Target values, strictly positive.

        Returns
        -------
        valid_centers : list of list[float]
            Median covariate value for each valid window, shape (n_valid, 1).
        mu_estimates : list of float
        sigma_estimates : list of float
        """
        n = len(y)
        sort_idx = np.argsort(x)
        x_sorted = x[sort_idx]
        y_sorted = y[sort_idx]

        if n <= self.bin_size:
            mu, sigma = self._robust_qq_estimation(y_sorted, self.lambda_)
            if mu is None:
                return [], [], []
            return [[float(np.median(x_sorted))]], [mu], [sigma]

        stride = max(1, (n - self.bin_size) // max(self.n_bins - 1, 1))
        valid_centers: List = []
        mu_estimates: List = []
        sigma_estimates: List = []

        for start in range(0, n - self.bin_size + 1, stride):
            bin_y = y_sorted[start : start + self.bin_size]
            bin_x = x_sorted[start : start + self.bin_size]
            if len(bin_y) < RobustNormalizerConfig.MIN_BIN_SAMPLES:
                continue
            mu, sigma = self._robust_qq_estimation(bin_y, self.lambda_)
            if mu is not None and sigma is not None:
                valid_centers.append([float(np.median(bin_x))])
                mu_estimates.append(mu)
                sigma_estimates.append(sigma)

        return valid_centers, mu_estimates, sigma_estimates

    def _create_knn_bins(self, X: np.ndarray, y: np.ndarray) -> Tuple[List, List, List]:
        """Build k-NN overlapping windows for a 2D continuous covariate space.

        Selects ``n_bins`` reference points by sorting data along a 1D projection
        (sum of scaled columns) and taking evenly-spaced rank positions.  For each
        reference point the ``bin_size`` nearest neighbours in the scaled 2D space
        form a bin.  This avoids grid-based binning and the curse of dimensionality.

        Parameters
        ----------
        X : ndarray of shape (n_samples, 2)
            Continuous covariate matrix (already log-transformed if applicable).
        y : ndarray of shape (n_samples,)
            Target values, strictly positive.

        Returns
        -------
        valid_centers : list of list[float]
            Median covariate values per valid window, shape (n_valid, 2).
        mu_estimates : list of float
        sigma_estimates : list of float
        """
        n = len(y)
        if n == 0:
            return [], [], []
        x_std = X.std(axis=0)
        x_std[x_std < 1e-8] = 1.0
        X_scaled = (X - X.mean(axis=0)) / x_std

        projection = X_scaled.sum(axis=1)
        proj_sorted_idx = np.argsort(projection)
        ref_positions = np.round(np.linspace(0, n - 1, self.n_bins)).astype(int)
        ref_idx = np.unique(proj_sorted_idx[ref_positions])

        k = min(self.bin_size, n)
        tree = KDTree(X_scaled)

        valid_centers: List = []
        mu_estimates: List = []
        sigma_estimates: List = []

        for idx in ref_idx:
            neighbor_idx = tree.query(X_scaled[[idx]], k=k, return_distance=False)[0]
            bin_y = y[neighbor_idx]
            bin_X = X[neighbor_idx]
            if len(bin_y) < RobustNormalizerConfig.MIN_BIN_SAMPLES:
                continue
            mu, sigma = self._robust_qq_estimation(bin_y, self.lambda_)
            if mu is not None and sigma is not None:
                valid_centers.append(
                    [float(np.median(bin_X[:, 0])), float(np.median(bin_X[:, 1]))]
                )
                mu_estimates.append(mu)
                sigma_estimates.append(sigma)

        return valid_centers, mu_estimates, sigma_estimates

    def _robust_qq_estimation(
        self, bin_data: np.ndarray, lambda_: float
    ) -> Tuple[Optional[float], Optional[float]]:
        """Estimate mu and sigma via Q-Q regression on Box-Cox transformed data.

        No outlier filtering is performed here; the caller is responsible for
        passing pre-filtered data. The Blom plotting position formula is applied
        to all n samples (FIX 2: positions computed for the actual count, not a
        pre-filter count).

        Parameters
        ----------
        bin_data : ndarray of shape (n_samples,)
            Target values for a single bin. Must be strictly positive.
        lambda_ : float
            Box-Cox transformation parameter.

        Returns
        -------
        mu : float or None
            Estimated location in Box-Cox space. ``None`` if estimation
            is not possible.
        sigma : float or None
            Estimated scale in Box-Cox space, floored at ``1e-6``. ``None``
            if estimation is not possible.
        """
        bin_data_bc = np.sort(self._transform(bin_data, lambda_))
        n = len(bin_data_bc)
        if n < 2:
            return None, None

        probs = (np.arange(1, n + 1) - 3 / 8) / (n + 1 / 4)
        z_theoretical = stats.norm.ppf(probs)

        if np.var(z_theoretical) < 1e-8:
            return None, None

        sigma, mu = np.polyfit(z_theoretical, bin_data_bc, deg=1)
        return mu, max(sigma, 1e-6)

    def _fit_fallback(self, y: np.ndarray) -> None:
        """Set global mu/sigma (in Box-Cox space) when binning is not viable.

        Parameters
        ----------
        y : ndarray of shape (n_samples,)
            Target values for the current group. Must be strictly positive.
        """
        mu, sigma = self._robust_qq_estimation(y, self.lambda_)
        if mu is None or sigma is None:
            y_bc = self._transform(y, self.lambda_)
            self._global_mu = float(np.mean(y_bc))
            self._global_sigma = max(float(np.std(y_bc)), 1e-6)
        else:
            self._global_mu = mu
            self._global_sigma = sigma
        self._is_fitted = False


__all__ = [
    "RobustNormalizerConfig",
    "ContinuousSurfaceFitter",
]
