import warnings
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple, Dict, List, Union
import numpy as np
import scipy.stats as stats
from scipy.stats import boxcox
from sklearn.base import BaseEstimator, TransformerMixin
from math import comb
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
        Box-Cox transformation parameter. When ``None`` the optimal value
        is found via a grid search over [-2, 2] that maximises the Pearson
        correlation between sorted BoxCox(y) values and their theoretical
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
    empirical quantiles of Box-Cox transformed window data against the
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
        bin_size: int = 120,
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
                valid_centers, mu_estimates, sigma_estimates = self._create_rolling_bins(
                    X_work[:, 0], y_work
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

            X_poly_work = self.poly_transformer.transform(X_work)
            mu_pred = self.mu_model.predict(X_poly_work)
            sigma_pred = np.maximum(self.sigma_model.predict(X_poly_work), 1e-6)
            y_bc = boxcox(y_work, lmbda=self.lambda_)
            z = (y_bc - mu_pred) / sigma_pred
            mask = np.abs(z) <= 3.372

            if mask.all():
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
                y_bc = np.sort(boxcox(y, lmbda=lam))
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

    def _create_knn_bins(
        self, X: np.ndarray, y: np.ndarray
    ) -> Tuple[List, List, List]:
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
        bin_data_bc = np.sort(boxcox(bin_data, lmbda=lambda_))
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
            y_bc = boxcox(y, lmbda=self.lambda_)
            self._global_mu = float(np.mean(y_bc))
            self._global_sigma = max(float(np.std(y_bc)), 1e-6)
        else:
            self._global_mu = mu
            self._global_sigma = sigma
        self._is_fitted = False


class RobustConditionalNormalizer(BaseEstimator, TransformerMixin):
    """Robust conditional Z-score normalization over categorical and continuous covariates.

    A single polynomial surface is fitted across the entire dataset (ignoring
    categorical groups during curve fitting). Base Z-scores are then computed
    for all samples using this shared surface. A post-hoc correction is applied
    per categorical group: the mean (``mu_cat``) and standard deviation
    (``sigma_cat``) of the base Z-scores within each group are subtracted and
    divided out, yielding ``Z_corrected = (Z_base - mu_cat) / sigma_cat``. This
    avoids the severe overfitting that arises from fitting independent surfaces
    per group.

    Outlier-robust estimation (iterative conditional Z-score fence + Q-Q
    regression on Box-Cox transformed values) is applied within each rolling
    window, making the normalizer resistant to heavy-tailed distributions common
    in flow cytometry and single-cell measurements.

    Parameters
    ----------
    categorical_cols : sequence of int
        Column indices of categorical covariates used to compute post-hoc
        group corrections (e.g. sex, batch). At most
        ``RobustNormalizerConfig.MAX_CATEGORICAL`` (default 2) columns.
    continuous_cols : sequence of int
        Column indices of continuous covariates used to model within-group
        variation of mu and sigma (e.g. age, BMI). Up to two columns are
        supported; one column uses 1D rolling windows, two columns use k-NN
        windows (``RobustNormalizerConfig.MAX_CONTINUOUS == 2``).
        When two columns are used, set ``n_bins >= 10`` so that the number
        of valid centers meets the minimum for a degree-3 surface polynomial
        (``comb(3+2, 2) == 10`` terms).
    target_col : int or "all"
        Column index of the marker to be normalised, or the string ``"all"``
        to normalise every column that is not listed in ``categorical_cols``
        or ``continuous_cols``. When ``"all"``, the set of target columns is
        determined at :meth:`fit` time from the width of the input matrix.
    n_bins : int, default=6
        Target number of rolling windows used to estimate the polynomial curve.
    degree : int, default=3
        Degree of the polynomial curve fitted over the window mu/sigma
        estimates. Mørkved et al. (2015) use cubic polynomials.
    n_iterations : int, default=3
        Maximum number of iterative conditional outlier-removal passes applied
        before the polynomial is finalised.
    log_transform_continuous : bool, default=False
        When ``True``, ``log10`` is applied to continuous covariate columns
        before fitting and prediction. Requires strictly positive values.
    bin_size : int, default=120
        Number of samples per rolling window. Mørkved et al. (2015) use 120.

    Attributes
    ----------
    _fitters : dict of int -> ContinuousSurfaceFitter
        One polynomial surface per target column, populated by :meth:`fit`.
    _cat_corrections : dict of int -> (dict of tuple -> (float, float))
        Per-target-column mapping from categorical combination to
        ``(mu_cat, sigma_cat)`` correction parameters.
    _resolved_target_cols : list of int
        The actual column indices to normalise, resolved from ``target_col``
        during :meth:`fit`.

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
        target_col: Union[int, str],
        n_bins: int = 6,
        degree: int = 3,
        n_iterations: int = 3,
        log_transform_continuous: bool = False,
        bin_size: int = 120,
    ):
        self.categorical_cols = categorical_cols
        self.continuous_cols = continuous_cols
        self.target_col = target_col
        self.n_bins = n_bins
        self.degree = degree
        self.n_iterations = n_iterations
        self.log_transform_continuous = log_transform_continuous
        self.bin_size = bin_size
        self._fitters: Dict[int, ContinuousSurfaceFitter] = {}
        self._cat_corrections: Dict[int, Dict[Tuple, Tuple[float, float]]] = {}
        self._cat_encoders: Dict[int, Dict] = {}
        self._resolved_target_cols: List[int] = []
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
        if self.target_col != "all":
            if not isinstance(self.target_col, int):
                raise ValueError(
                    "target_col must be an integer column index or the string 'all'."
                )
            covariate_cols = list(self.categorical_cols) + list(self.continuous_cols)
            if self.target_col in covariate_cols:
                raise ValueError("Target column cannot be included in covariates.")

    def _encode(self, X: np.ndarray, fit: bool = False) -> np.ndarray:
        X = np.asarray(X)
        if X.dtype.kind in "iufcb":
            return X.astype(float)
        if fit:
            self._cat_encoders = {
                c: {v: float(i) for i, v in enumerate(sorted(set(X[:, c]), key=str))}
                for c in self.categorical_cols
            }
        out = np.empty(X.shape, dtype=float)
        for j in range(X.shape[1]):
            enc = self._cat_encoders.get(j)
            out[:, j] = [enc[v] for v in X[:, j]] if enc else X[:, j].astype(float)
        return out

    def _resolve_target_cols(self, n_cols: int) -> List[int]:
        covariate_cols = set(self.categorical_cols) | set(self.continuous_cols)
        if self.target_col == "all":
            resolved = [c for c in range(n_cols) if c not in covariate_cols]
            if not resolved:
                raise ValueError("No target columns remain after excluding covariates.")
            return resolved
        return [int(self.target_col)]

    def fit(
        self, X: np.ndarray, y: Optional[np.ndarray] = None
    ) -> "RobustConditionalNormalizer":
        """Fit a single combined surface and compute per-group categorical corrections.

        A single :class:`ContinuousSurfaceFitter` is fitted on the full dataset,
        ignoring categorical group membership during curve fitting. Base Z-scores
        are computed for every sample, then grouped by categorical combination to
        derive ``(mu_cat, sigma_cat)`` correction parameters stored in
        ``_cat_corrections``.

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
        X = self._encode(X, fit=True)
        self._resolved_target_cols = self._resolve_target_cols(X.shape[1])

        X_cont_all = X[:, list(self.continuous_cols)]

        if self.categorical_cols:
            cat_data = X[:, list(self.categorical_cols)]
            unique_rows, inverse_indices = np.unique(
                cat_data, axis=0, return_inverse=True
            )
        else:
            unique_rows = inverse_indices = None

        self._fitters = {}
        self._cat_corrections = {}

        for col in self._resolved_target_cols:
            y_all = X[:, col]

            fitter = ContinuousSurfaceFitter(
                n_bins=self.n_bins,
                degree=self.degree,
                n_iterations=self.n_iterations,
                log_transform_continuous=self.log_transform_continuous,
                bin_size=self.bin_size,
            )
            fitter.fit(X_cont_all, y_all)
            self._fitters[col] = fitter

            y_bc_all = boxcox(y_all, lmbda=fitter.lambda_)
            mu_all, sigma_all = fitter.predict_mu_sigma(X_cont_all)
            z_base_all = (y_bc_all - mu_all) / np.maximum(sigma_all, 1e-6)

            col_corrections: Dict[Tuple, Tuple[float, float]] = {}
            if not self.categorical_cols:
                col_corrections[tuple()] = (0.0, 1.0)
            else:
                for i in range(len(unique_rows)):
                    cat_tuple = tuple(unique_rows[i])
                    z_group = z_base_all[inverse_indices == i]
                    mu_cat = float(np.mean(z_group))
                    sigma_cat = max(float(np.std(z_group)), 1e-6)
                    col_corrections[cat_tuple] = (mu_cat, sigma_cat)
            self._cat_corrections[col] = col_corrections

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply conditional Z-score normalization with post-hoc categorical correction.

        Each sample's base Z-score is computed as
        ``(BoxCox(y) - mu(x)) / sigma(x)`` using the shared polynomial surface,
        then corrected for its categorical group:
        ``Z_corrected = (Z_base - mu_cat) / sigma_cat``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_cols)
            Input matrix with the same column layout as used in :meth:`fit`.
            Target values must be strictly positive (required by Box-Cox).

        Returns
        -------
        X_out : ndarray of shape (n_samples, n_cols)
            Copy of ``X`` with ``target_col`` replaced by corrected Z-scores.
            All other columns are unchanged.

        Warns
        -----
        UserWarning
            Raised for any categorical combination not seen during
            :meth:`fit`. Affected samples receive a Z-score of 0.

        Raises
        ------
        ValueError
            If any target value is <= 0.
        """
        X_out = self._encode(X)
        X_cont = X_out[:, list(self.continuous_cols)]

        if not self.categorical_cols:
            cat_groups = [(tuple(), np.arange(X_out.shape[0]))]
        else:
            cat_data = X_out[:, list(self.categorical_cols)]
            unique_rows, inverse_indices = np.unique(
                cat_data, axis=0, return_inverse=True
            )
            cat_groups = [
                (tuple(unique_rows[i]), np.where(inverse_indices == i)[0])
                for i in range(len(unique_rows))
            ]

        for col in self._resolved_target_cols:
            y_raw = X_out[:, col]

            if np.any(y_raw <= 0):
                raise ValueError(
                    f"Column {col} contains values <= 0; Box-Cox transform requires "
                    "strictly positive data."
                )

            fitter = self._fitters[col]
            y_bc = boxcox(y_raw, lmbda=fitter.lambda_)
            mu_pred, sigma_pred = fitter.predict_mu_sigma(X_cont)
            z_base = (y_bc - mu_pred) / np.maximum(sigma_pred, 1e-6)

            col_corrections = self._cat_corrections[col]
            for cat_tuple, row_indices in cat_groups:
                if cat_tuple not in col_corrections:
                    warnings.warn(
                        f"Unseen categorical combination {cat_tuple} for column {col}. "
                        "Setting Z-scores to 0."
                    )
                    X_out[row_indices, col] = 0.0
                    continue
                mu_cat, sigma_cat = col_corrections[cat_tuple]
                X_out[row_indices, col] = (z_base[row_indices] - mu_cat) / sigma_cat

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
