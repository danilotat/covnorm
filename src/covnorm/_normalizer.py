import warnings
from dataclasses import dataclass
from itertools import product
from typing import Optional, Sequence, Tuple, Dict

import numpy as np
import scipy.stats as stats
from scipy.stats import boxcox, boxcox_normmax
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
        Mørkved et al. (2015) use ~120 per bin; 30 is the practical
        minimum for a generalised implementation.
    """

    MAX_CATEGORICAL: int = 2
    MAX_CONTINUOUS: int = 2
    MIN_BIN_SAMPLES: int = 30


class ContinuousSurfaceFitter:
    """Polynomial surface model for mu and sigma over continuous covariates.

    For each categorical subgroup, this class partitions the continuous
    covariate space into a quantile-based grid of equal-size bins, estimates
    mu and sigma per bin via Q-Q regression on Box-Cox transformed data with
    iterative Tukey z-score outlier rejection, then fits a polynomial surface
    through the valid bin estimates. At prediction time the surface is
    evaluated at the exact covariate values of each sample.

    When no continuous covariates are present, or when the number of valid
    bins is insufficient for stable polynomial fitting, the estimator falls
    back to a single global (mu, sigma) pair derived from the full group.
    All mu and sigma values live in Box-Cox transformed space.

    Parameters
    ----------
    n_bins : int
        Number of equal-size bins per continuous covariate axis.
    degree : int, default=3
        Degree of the polynomial surface fitted to the bin estimates.
        Mørkved et al. (2015) use cubic polynomials.
    lambda_ : float or None, default=None
        Box-Cox transformation parameter. When ``None`` the optimal value
        is found automatically via ``scipy.stats.boxcox_normmax`` during
        :meth:`fit` and stored as ``self.lambda_``. When provided, that
        value is used directly and stored unchanged.
    n_iterations : int, default=3
        Number of iterative Tukey z-score outlier-removal passes applied
        to the full group data before binning.
    log_transform_continuous : bool, default=False
        When ``True``, ``log10`` is applied to all continuous covariate
        columns at the start of :meth:`fit` and :meth:`predict_mu_sigma`.
        Requires all covariate values to be strictly positive.

    Attributes
    ----------
    poly_transformer : sklearn.preprocessing.PolynomialFeatures
        Transforms covariate vectors into polynomial feature matrices.
    mu_model : sklearn.linear_model.LinearRegression
        Linear model predicting mu (in Box-Cox space) from polynomial features.
    sigma_model : sklearn.linear_model.LinearRegression
        Linear model predicting sigma (in Box-Cox space) from polynomial features.
    lambda_ : float
        Box-Cox parameter; ``None`` until :meth:`fit` is called.
    _global_mu : float
        Fallback mu (in Box-Cox space) used when the polynomial surface
        cannot be fitted.
    _global_sigma : float
        Fallback sigma (in Box-Cox space) used when the polynomial surface
        cannot be fitted.
    _is_fitted : bool
        ``True`` if the polynomial surface was successfully fitted,
        ``False`` if the fallback values are in use.

    Notes
    -----
    Q-Q regression estimates mu and sigma by fitting a line through the
    empirical quantiles of Box-Cox transformed bin data against the
    corresponding theoretical normal quantiles using the Blom plotting
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
    ):
        self.n_bins = n_bins
        self.degree = degree
        self.lambda_ = lambda_
        self.n_iterations = n_iterations
        self.log_transform_continuous = log_transform_continuous
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
            All values must be strictly positive when
            ``log_transform_continuous=True``.
        y : ndarray of shape (n_samples,)
            Target values for the current categorical subgroup. Must be
            strictly positive (required by the Box-Cox transform).

        Returns
        -------
        self : ContinuousSurfaceFitter

        Raises
        ------
        ValueError
            If any value in ``y`` is <= 0, or if ``log_transform_continuous``
            is ``True`` and any continuous covariate value is <= 0.
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

        if np.any(y <= 0):
            raise ValueError(
                "y contains values <= 0; Box-Cox transform requires strictly "
                "positive data. Shift the data before fitting."
            )

        if self.lambda_ is None:
            self.lambda_ = float(boxcox_normmax(y))

        y_clean = y.copy()
        X_cont_clean = X_cont.copy()
        for _ in range(self.n_iterations):
            mask = self._tukey_z_filter(y_clean)
            if mask.all():
                break
            y_clean = y_clean[mask]
            if n_features > 0:
                X_cont_clean = X_cont_clean[mask]

        if n_features == 0:
            self._fit_fallback(y_clean)
            return self

        n_clean = len(y_clean)
        # equal size bins. Note that this diverges from the previous implementation where bins where computed using np.linspace, thus not quantiles.
        bin_edges = [
            np.quantile(X_cont_clean[:, c], np.linspace(0, 1, self.n_bins + 1))
            for c in range(n_features)
        ]
        valid_centers, mu_estimates, sigma_estimates = [], [], []
        # that's the core of covnorm. Here we're doing iteratively the computation for each bins in each feature.
        for bin_indices in product(*(range(self.n_bins) for _ in range(n_features))):
            mask = np.ones(n_clean, dtype=bool)
            centers = []
            for c, b_idx in enumerate(bin_indices):
                lower, upper = bin_edges[c][b_idx], bin_edges[c][b_idx + 1]
                if b_idx == self.n_bins - 1:
                    mask &= (X_cont_clean[:, c] >= lower) & (
                        X_cont_clean[:, c] <= upper
                    )
                else:
                    mask &= (X_cont_clean[:, c] >= lower) & (X_cont_clean[:, c] < upper)
                centers.append((lower + upper) / 2.0)

            bin_y = y_clean[mask]
            if len(bin_y) >= RobustNormalizerConfig.MIN_BIN_SAMPLES:
                mu, sigma = self._robust_qq_estimation(bin_y, self.lambda_)
                if mu is not None and sigma is not None:
                    valid_centers.append(centers)
                    mu_estimates.append(mu)
                    sigma_estimates.append(sigma)

        if len(valid_centers) < (self.degree + 1) ** n_features:
            warnings.warn(
                "Too few valid bins for stable polynomial fitting. Falling back to global estimates."
            )
            self._fit_fallback(y_clean)
            return self

        X_poly = self.poly_transformer.fit_transform(np.array(valid_centers))
        self.mu_model.fit(X_poly, mu_estimates)
        self.sigma_model.fit(X_poly, sigma_estimates)
        self._is_fitted = True
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
        # NOTE: this log10 transform is used to match the original paper implementation. Here, to ensure flexibility is leaved as an argument, but in most of the cases it should not be required
        if self.log_transform_continuous:
            X_cont = np.log10(X_cont)
        X_poly = self.poly_transformer.transform(X_cont)
        return self.mu_model.predict(X_poly), self.sigma_model.predict(X_poly)

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

        Raises
        ------
        ValueError
            If any value in ``bin_data`` is <= 0.
        """
        if np.any(bin_data <= 0):
            raise ValueError(
                "bin_data contains values <= 0; shift the data before "
                "applying Box-Cox."
            )
        # this apply box-cox transforming before doing sort-regression
        bin_data_bc = np.sort(boxcox(bin_data, lmbda=lambda_))
        n = len(bin_data_bc)
        if n < 2:
            return None, None

        # NOTE: blom formula for computing theoretically probs, based on the gaussian assumption. While this is for keeping concordance with the original publication, would be great to eval its power and consider deviations to this, maybe with other background distributions.
        probs = (np.arange(1, n + 1) - 3 / 8) / (n + 1 / 4)
        z_theoretical = stats.norm.ppf(probs)

        if np.var(z_theoretical) < 1e-8:
            return None, None

        sigma, mu = np.polyfit(z_theoretical, bin_data_bc, deg=1)
        return mu, max(sigma, 1e-6)

    def _tukey_z_filter(self, y: np.ndarray, z_threshold: float = 3.372) -> np.ndarray:
        """Identify inliers using a z-score threshold in Box-Cox transformed space.

        Applies Tukey's fence with factor 2 in z-score space, equivalent to
        |z| > 3.372 for a standard normal distribution (Mørkved et al. 2015,
        § Methods). A preliminary mu and sigma are obtained from
        :meth:`_robust_qq_estimation` and used to compute per-sample z-scores.

        Parameters
        ----------
        y : ndarray of shape (n_samples,)
            Raw target values. Must be strictly positive.
        z_threshold : float, default=3.372
            Samples with ``|z| > z_threshold`` in Box-Cox space are flagged
            as outliers. The default corresponds to Tukey's factor 2 in
            z-score space.

        Returns
        -------
        mask : ndarray of bool, shape (n_samples,)
            ``True`` for inlier samples.
        """
        mu, sigma = self._robust_qq_estimation(y, self.lambda_)
        if mu is None:
            return np.ones(len(y), dtype=bool)
        y_bc = boxcox(y, lmbda=self.lambda_)
        z = (y_bc - mu) / sigma
        return np.abs(z) <= z_threshold

    def _fit_fallback(self, y: np.ndarray) -> None:
        """Set global mu/sigma (in Box-Cox space) when binning is not viable.

        Parameters
        ----------
        y : ndarray of shape (n_samples,)
            Target values for the current categorical subgroup. Must be
            strictly positive.
        """
        mu, sigma = self._robust_qq_estimation(y, self.lambda_)
        if mu is None or sigma is None:
            y_bc = boxcox(y, lmbda=self.lambda_)
            self._global_mu = float(np.mean(y_bc))
            self._global_sigma = max(float(np.std(y_bc)), 1e-6)
        else:
            self._global_mu = mu
            self._global_sigma = sigma
        self._is_fitted = False


class RobustConditionalNormalizer(BaseEstimator, TransformerMixin):
    """Robust conditional Z-score normalization over categorical and continuous covariates.

    For each unique combination of categorical covariate values the
    transformer fits a :class:`ContinuousSurfaceFitter` that models how the
    location (mu) and scale (sigma) of the target marker vary across the
    continuous covariate space. At transform time, each sample is
    standardised as ``z = (BoxCox(y) - mu(x)) / sigma(x)`` where ``x`` is
    its vector of continuous covariate values and mu/sigma live in Box-Cox
    transformed space.

    Outlier-robust estimation (iterative Tukey z-score fence + Q-Q regression
    on Box-Cox transformed values) is applied within each bin, making the
    normalizer resistant to heavy-tailed distributions common in flow
    cytometry and single-cell measurements.

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
        Number of equal-size bins per continuous covariate axis used to
        estimate the polynomial surface.
    degree : int, default=3
        Degree of the polynomial surface fitted over the binned mu/sigma
        estimates. Mørkved et al. (2015) use cubic polynomials.
    n_iterations : int, default=3
        Number of iterative Tukey z-score outlier-removal passes applied
        to each group's data before binning.
    log_transform_continuous : bool, default=False
        When ``True``, ``log10`` is applied to continuous covariate columns
        before fitting and prediction. Requires strictly positive values.

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
    ...     rng.integers(0, 2, 500).astype(float),  # sex (col 0, categorical)
    ...     rng.uniform(1, 80, 500),                 # age (col 1, continuous)
    ...     rng.gamma(2, 10, 500),                   # marker (col 2, target)
    ... ])
    >>> norm = RobustConditionalNormalizer(
    ...     categorical_cols=[0],
    ...     continuous_cols=[1],
    ...     target_col=2,
    ...     log_transform_continuous=True,
    ... )
    >>> X_norm = norm.fit_transform(X)
    >>> X_norm.shape
    (500, 3)
    """

    def __init__(
        self,
        categorical_cols: Sequence[int],
        continuous_cols: Sequence[int],
        target_col: int,
        n_bins: int = 6,
        degree: int = 3,
        n_iterations: int = 3,
        log_transform_continuous: bool = False,
    ):
        self.categorical_cols = categorical_cols
        self.continuous_cols = continuous_cols
        self.target_col = target_col
        self.n_bins = n_bins
        self.degree = degree
        self.n_iterations = n_iterations
        self.log_transform_continuous = log_transform_continuous
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
            Target values must be strictly positive (required by Box-Cox).
        y : ignored
            Not used. Present for scikit-learn pipeline compatibility.

        Returns
        -------
        self : RobustConditionalNormalizer
            Fitted estimator.
        """
        X = np.asarray(X, dtype=float)
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
            fitter = ContinuousSurfaceFitter(
                n_bins=self.n_bins,
                degree=self.degree,
                n_iterations=self.n_iterations,
                log_transform_continuous=self.log_transform_continuous,
            )
            fitter.fit(X_cont, y_target)
            self._models[cat_tuple] = fitter

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply conditional Z-score normalization to the target column.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_cols)
            Input matrix with the same column layout as used in :meth:`fit`.
            Target values must be strictly positive (required by Box-Cox).

        Returns
        -------
        X_out : ndarray of shape (n_samples, n_cols)
            Copy of ``X`` with ``target_col`` replaced by Z-scores computed
            as ``(BoxCox(y) - mu(x)) / sigma(x)`` where mu and sigma are in
            Box-Cox transformed space. All other columns are unchanged.

        Warns
        -----
        UserWarning
            Raised for any categorical combination not seen during
            :meth:`fit`. Affected samples receive a Z-score of 0.

        Raises
        ------
        ValueError
            If any target value is <= 0 for a known categorical group.
        """
        X_out = np.copy(np.asarray(X, dtype=float))

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

            # FIX 3e: apply Box-Cox to y_raw before computing z-scores
            if np.any(y_raw <= 0):
                raise ValueError(
                    f"Target values for categorical group {cat_tuple} contain "
                    "values <= 0; Box-Cox transform requires strictly positive data."
                )
            y_transformed = boxcox(y_raw, lmbda=fitter.lambda_)

            mu_pred, sigma_pred = fitter.predict_mu_sigma(X_cont)
            sigma_pred = np.maximum(sigma_pred, 1e-6)
            X_out[row_indices, self.target_col] = (y_transformed - mu_pred) / sigma_pred

        return X_out


__all__ = [
    "RobustNormalizerConfig",
    "ContinuousSurfaceFitter",
    "RobustConditionalNormalizer",
]


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    n_per_group = 1000

    groups = []
    for group_id in (0, 1):
        age = rng.uniform(0.01, 100, n_per_group)
        marker = rng.gamma(
            shape=2, scale=10, size=n_per_group
        )  # right-skewed, positive
        cat = np.full(n_per_group, float(group_id))
        groups.append(np.column_stack([cat, age, marker]))
    X = np.vstack(groups)

    norm = RobustConditionalNormalizer(
        categorical_cols=[0],
        continuous_cols=[1],
        target_col=2,
        log_transform_continuous=True,
    )
    X_out = norm.fit_transform(X)

    assert X_out.shape == X.shape, f"Shape mismatch: {X_out.shape} != {X.shape}"

    for group_id in (0, 1):
        mask = X_out[:, 0] == float(group_id)
        z = X_out[mask, 2]
        mean_z = float(np.mean(z))
        std_z = float(np.std(z))
        assert (
            abs(mean_z) < 0.1
        ), f"Group {group_id}: mean z = {mean_z:.4f}, expected |mean| < 0.1"
        assert (
            abs(std_z - 1.0) < 0.2
        ), f"Group {group_id}: std z = {std_z:.4f}, expected |std - 1| < 0.2"

    print("PASS")
