import warnings
from dataclasses import dataclass
from itertools import product
from typing import Optional, Sequence, Tuple, Dict

import numpy as np
import scipy.stats as stats
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LinearRegression
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
        :class:`RobustConditionalNormalizer`.
    MIN_BIN_SAMPLES : int
        Minimum number of samples required in a bin to attempt
        Q-Q regression. Bins below this threshold are skipped.
    """

    MAX_CATEGORICAL: int = 2
    MAX_CONTINUOUS: int = 2
    MIN_BIN_SAMPLES: int = 5


class ContinuousSurfaceFitter:
    """Polynomial surface model for mu and sigma over continuous covariates.

    For each categorical subgroup, this class partitions the continuous
    covariate space into a regular grid of bins, estimates mu and sigma
    per bin via Q-Q regression with Tukey's fence outlier rejection, then
    fits a polynomial surface through the valid bin estimates. At prediction
    time the surface is evaluated at the exact covariate values of each
    sample.

    When no continuous covariates are present, or when the number of valid
    bins is insufficient for stable polynomial fitting, the estimator falls
    back to a single global (mu, sigma) pair derived from the full group.

    Parameters
    ----------
    n_bins : int
        Number of equal-width bins per continuous covariate axis.
    degree : int
        Degree of the polynomial surface fitted to the bin estimates.

    Attributes
    ----------
    poly_transformer : sklearn.preprocessing.PolynomialFeatures
        Transforms covariate vectors into polynomial feature matrices.
    mu_model : sklearn.linear_model.LinearRegression
        Linear model predicting mu from polynomial features.
    sigma_model : sklearn.linear_model.LinearRegression
        Linear model predicting sigma from polynomial features.
    _global_mu : float
        Fallback mu used when the polynomial surface cannot be fitted.
    _global_sigma : float
        Fallback sigma used when the polynomial surface cannot be fitted.
    _is_fitted : bool
        ``True`` if the polynomial surface was successfully fitted,
        ``False`` if the fallback values are in use.

    Notes
    -----
    Q-Q regression estimates mu and sigma by fitting a line through the
    empirical quantiles of the (outlier-filtered) bin data against the
    corresponding theoretical normal quantiles. The slope gives sigma and
    the intercept gives mu. Sigma is floored at ``1e-6`` to prevent
    division by zero during Z-score computation.
    """

    def __init__(self, n_bins: int, degree: int):
        self.n_bins = n_bins
        self.degree = degree
        self.poly_transformer = PolynomialFeatures(
            degree=self.degree, include_bias=True
        )
        self.mu_model = LinearRegression(fit_intercept=False)
        self.sigma_model = LinearRegression(fit_intercept=False)
        self._global_mu: float = 0.0
        self._global_sigma: float = 1.0
        self._is_fitted: bool = False

    def fit(self, X_cont: np.ndarray, y: np.ndarray) -> "ContinuousSurfaceFitter":
        """Fit the polynomial mu/sigma surface to binned estimates.

        Parameters
        ----------
        X_cont : ndarray of shape (n_samples, n_features)
            Continuous covariate matrix. Pass an array with zero columns
            (``n_features == 0``) to trigger the global fallback.
        y : ndarray of shape (n_samples,)
            Target values for the current categorical subgroup.

        Returns
        -------
        self : ContinuousSurfaceFitter
        """
        n_samples, n_features = X_cont.shape
        if n_features == 0:
            self._fit_fallback(y)
            return self

        bin_edges = [
            np.linspace(X_cont[:, c].min(), X_cont[:, c].max(), self.n_bins + 1)
            for c in range(n_features)
        ]
        valid_centers, mu_estimates, sigma_estimates = [], [], []

        for bin_indices in product(*(range(self.n_bins) for _ in range(n_features))):
            mask = np.ones(n_samples, dtype=bool)
            centers = []
            for c, b_idx in enumerate(bin_indices):
                lower, upper = bin_edges[c][b_idx], bin_edges[c][b_idx + 1]
                if b_idx == self.n_bins - 1:
                    mask &= (X_cont[:, c] >= lower) & (X_cont[:, c] <= upper)
                else:
                    mask &= (X_cont[:, c] >= lower) & (X_cont[:, c] < upper)
                centers.append((lower + upper) / 2.0)

            bin_y = y[mask]
            if len(bin_y) >= RobustNormalizerConfig.MIN_BIN_SAMPLES:
                mu, sigma = self._robust_qq_estimation(bin_y)
                if mu is not None and sigma is not None:
                    valid_centers.append(centers)
                    mu_estimates.append(mu)
                    sigma_estimates.append(sigma)

        if len(valid_centers) < (self.degree + 1) ** n_features:
            warnings.warn(
                "Too few valid bins for stable polynomial fitting. Falling back to global estimates."
            )
            self._fit_fallback(y)
            return self

        X_poly = self.poly_transformer.fit_transform(np.array(valid_centers))
        self.mu_model.fit(X_poly, mu_estimates)
        self.sigma_model.fit(X_poly, sigma_estimates)
        self._is_fitted = True
        return self

    def predict_mu_sigma(self, X_cont: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict mu and sigma for each sample from its continuous covariates.

        Parameters
        ----------
        X_cont : ndarray of shape (n_samples, n_features)
            Continuous covariate matrix. Must have the same number of
            columns as the array passed to :meth:`fit`.

        Returns
        -------
        mu : ndarray of shape (n_samples,)
            Predicted location parameter for each sample.
        sigma : ndarray of shape (n_samples,)
            Predicted scale parameter for each sample. Always positive.
        """
        if not self._is_fitted or X_cont.shape[1] == 0:
            return (
                np.full(X_cont.shape[0], self._global_mu),
                np.full(X_cont.shape[0], self._global_sigma),
            )
        X_poly = self.poly_transformer.transform(X_cont)
        return self.mu_model.predict(X_poly), self.sigma_model.predict(X_poly)

    def _robust_qq_estimation(
        self, bin_data: np.ndarray
    ) -> Tuple[Optional[float], Optional[float]]:
        """Estimate mu and sigma via Q-Q regression with Tukey's fence filtering.

        Parameters
        ----------
        bin_data : ndarray of shape (n_samples,)
            Raw target values for a single bin.

        Returns
        -------
        mu : float or None
            Estimated location. ``None`` if estimation is not possible.
        sigma : float or None
            Estimated scale, floored at ``1e-6``. ``None`` if estimation
            is not possible.
        """
        bin_data = np.sort(bin_data)
        Q1, Q3 = np.percentile(bin_data, [25, 75])
        IQR = Q3 - Q1
        lower_bound, upper_bound = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR

        valid_mask = (bin_data >= lower_bound) & (bin_data <= upper_bound)
        valid_data = bin_data[valid_mask]
        if len(valid_data) < 2:
            return None, None

        n = len(bin_data)
        probs = (np.arange(1, n + 1) - 0.5) / n
        z_theoretical = stats.norm.ppf(probs)[valid_mask]

        if np.var(z_theoretical) < 1e-8:
            return None, None

        sigma, mu = np.polyfit(z_theoretical, valid_data, deg=1)
        return mu, max(sigma, 1e-6)

    def _fit_fallback(self, y: np.ndarray) -> None:
        """Set global mu/sigma from the full group when binning is not viable.

        Parameters
        ----------
        y : ndarray of shape (n_samples,)
            Target values for the current categorical subgroup.
        """
        mu, sigma = self._robust_qq_estimation(y)
        self._global_mu = mu if mu is not None else float(np.mean(y))
        self._global_sigma = sigma if sigma is not None else max(float(np.std(y)), 1e-6)
        self._is_fitted = False


class RobustConditionalNormalizer(BaseEstimator, TransformerMixin):
    """Robust conditional Z-score normalization over categorical and continuous covariates.

    For each unique combination of categorical covariate values the
    transformer fits a :class:`ContinuousSurfaceFitter` that models how the
    location (mu) and scale (sigma) of the target marker vary across the
    continuous covariate space. At transform time, each sample is
    standardised as ``z = (y - mu(x)) / sigma(x)`` where ``x`` is its
    vector of continuous covariate values.

    Outlier-robust estimation (Tukey's fences + Q-Q regression) is applied
    within each bin, making the normalizer resistant to heavy-tailed
    distributions common in flow cytometry and single-cell measurements.

    Parameters
    ----------
    categorical_cols : sequence of int
        Column indices of categorical covariates used to partition samples
        into independent subgroups (e.g. sex, batch). At most
        ``RobustNormalizerConfig.MAX_CATEGORICAL`` (default 2) columns.
    continuous_cols : sequence of int
        Column indices of continuous covariates used to model within-group
        variation of mu and sigma (e.g. age, cell count). At most
        ``RobustNormalizerConfig.MAX_CONTINUOUS`` (default 2) columns.
    target_col : int
        Column index of the marker to be normalised. Must not appear in
        ``categorical_cols`` or ``continuous_cols``.
    n_bins : int, default=6
        Number of equal-width bins per continuous covariate axis used to
        estimate the polynomial surface.
    degree : int, default=2
        Degree of the polynomial surface fitted over the binned mu/sigma
        estimates.

    Attributes
    ----------
    _models : dict of tuple -> ContinuousSurfaceFitter
        Maps each observed categorical combination (as a tuple) to its
        fitted :class:`ContinuousSurfaceFitter`. Populated by :meth:`fit`.

    Notes
    -----
    ``X`` passed to :meth:`fit` and :meth:`transform` must be a 2-D array
    that contains both the covariate columns and the target column. The
    ``y`` argument of :meth:`fit` is ignored and exists only for
    scikit-learn pipeline compatibility.

    Unseen categorical combinations at transform time produce a warning and
    are assigned a Z-score of 0.

    Examples
    --------
    >>> import numpy as np
    >>> from covnorm import RobustConditionalNormalizer
    >>> rng = np.random.default_rng(0)
    >>> X = np.column_stack([
    ...     rng.integers(0, 2, 200),        # sex (col 0, categorical)
    ...     rng.uniform(20, 80, 200),        # age (col 1, continuous)
    ...     rng.normal(0, 1, 200),           # marker (col 2, target)
    ... ])
    >>> norm = RobustConditionalNormalizer(
    ...     categorical_cols=[0],
    ...     continuous_cols=[1],
    ...     target_col=2,
    ... )
    >>> X_norm = norm.fit_transform(X)
    >>> X_norm.shape
    (200, 3)
    """

    def __init__(
        self,
        categorical_cols: Sequence[int],
        continuous_cols: Sequence[int],
        target_col: int,
        n_bins: int = 6,
        degree: int = 2,
    ):
        self.categorical_cols = categorical_cols
        self.continuous_cols = continuous_cols
        self.target_col = target_col
        self.n_bins = n_bins
        self.degree = degree
        self._models: Dict[Tuple, ContinuousSurfaceFitter] = {}
        self._validate_constraints()

    def _validate_constraints(self) -> None:
        if len(self.categorical_cols) > RobustNormalizerConfig.MAX_CATEGORICAL:
            raise ValueError(
                f"Exceeded max categorical covariates ({RobustNormalizerConfig.MAX_CATEGORICAL})."
            )
        if len(self.continuous_cols) > RobustNormalizerConfig.MAX_CONTINUOUS:
            raise ValueError(
                f"Exceeded max continuous covariates ({RobustNormalizerConfig.MAX_CONTINUOUS})."
            )
        all_cols = list(self.categorical_cols) + list(self.continuous_cols)
        if self.target_col in all_cols:
            raise ValueError("Target column cannot be included in covariates.")

    def fit(
        self, X: np.ndarray, y: Optional[np.ndarray] = None
    ) -> "RobustConditionalNormalizer":
        """Fit one surface model per categorical subgroup.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_cols)
            Input matrix containing covariate columns and the target column.
        y : ignored
            Not used. Present for scikit-learn pipeline compatibility.

        Returns
        -------
        self : RobustConditionalNormalizer
            Fitted estimator.
        """
        X = np.asarray(X)
        self._models.clear()

        if not self.categorical_cols:
            cat_groups = [(tuple(), np.arange(X.shape[0]))]
        else:
            cat_data = X[:, self.categorical_cols]
            unique_rows, inverse_indices = np.unique(
                cat_data, axis=0, return_inverse=True
            )
            cat_groups = [
                (tuple(unique_rows[i]), np.where(inverse_indices == i)[0])
                for i in range(len(unique_rows))
            ]

        for cat_tuple, row_indices in cat_groups:
            X_cont = X[row_indices][:, self.continuous_cols]
            y_target = X[row_indices, self.target_col]
            fitter = ContinuousSurfaceFitter(n_bins=self.n_bins, degree=self.degree)
            fitter.fit(X_cont, y_target)
            self._models[cat_tuple] = fitter

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply conditional Z-score normalization to the target column.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_cols)
            Input matrix with the same column layout as used in :meth:`fit`.

        Returns
        -------
        X_out : ndarray of shape (n_samples, n_cols)
            Copy of ``X`` with ``target_col`` replaced by Z-scores.
            All other columns are unchanged.

        Warns
        -----
        UserWarning
            Raised for any categorical combination not seen during
            :meth:`fit`. Affected samples receive a Z-score of 0.
        """
        X_out = np.copy(np.asarray(X))

        if not self.categorical_cols:
            cat_groups = [(tuple(), np.arange(X_out.shape[0]))]
        else:
            cat_data = X_out[:, self.categorical_cols]
            unique_rows, inverse_indices = np.unique(
                cat_data, axis=0, return_inverse=True
            )
            cat_groups = [
                (tuple(unique_rows[i]), np.where(inverse_indices == i)[0])
                for i in range(len(unique_rows))
            ]

        for cat_tuple, row_indices in cat_groups:
            if cat_tuple not in self._models:
                warnings.warn(
                    f"Unseen categorical combination {cat_tuple}. Setting Z-scores to 0."
                )
                X_out[row_indices, self.target_col] = 0.0
                continue

            fitter = self._models[cat_tuple]
            X_cont = X_out[row_indices][:, self.continuous_cols]
            y_raw = X_out[row_indices, self.target_col]
            mu_pred, sigma_pred = fitter.predict_mu_sigma(X_cont)
            sigma_pred = np.maximum(sigma_pred, 1e-6)
            X_out[row_indices, self.target_col] = (y_raw - mu_pred) / sigma_pred

        return X_out
