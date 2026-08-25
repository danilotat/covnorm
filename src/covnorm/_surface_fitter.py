import warnings
from dataclasses import dataclass
from math import comb
from typing import List, Optional, Tuple

import numpy as np
import scipy.stats as stats
from scipy.stats import boxcox, yeojohnson
from sklearn.linear_model import Ridge
from sklearn.neighbors import KDTree
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

_ANCHOR_STRATEGIES = ("farthest_point", "projection_rank")
_CONTINUOUS_TRANSFORMS = (None, "log10", "zscore")
_SIGMA_FLOOR = 1e-6
_LOG_SIGMA_FLOOR = float(np.log(_SIGMA_FLOOR))
_ZSCORE_OUTLIER_THRESHOLD = 3.372


def _warn_log_transform_continuous_deprecated() -> None:
    warnings.warn(
        "log_transform_continuous is deprecated and will be removed in covnorm "
        "1.x; use transform_continuous='log10' instead.",
        FutureWarning,
        stacklevel=3,
    )


def _resolve_continuous_transform(
    transform_continuous: Optional[str], log_transform_continuous: bool
) -> Optional[str]:
    """Resolve the new transform option and its legacy boolean alias."""
    # TODO(v1.x): remove log_transform_continuous and this legacy resolution path.
    if transform_continuous not in _CONTINUOUS_TRANSFORMS:
        raise ValueError(
            "transform_continuous must be one of None, 'log10', or 'zscore'; "
            f"got {transform_continuous!r}."
        )
    if log_transform_continuous:
        if transform_continuous not in (None, "log10"):
            raise ValueError(
                "log_transform_continuous=True conflicts with "
                f"transform_continuous={transform_continuous!r}."
            )
        return "log10"
    return transform_continuous


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
    ridge_alpha : float, default=0.05
        L2 penalty relative to the mean squared fitting loss for both the mu and
        log-sigma surfaces. The value passed to scikit-learn is
        ``ridge_alpha * n_valid_bins``. Set to ``0.0`` to disable regularization.
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
        Legacy alias for ``transform_continuous='log10'``. Retained for
        backward compatibility and scheduled for removal in version 1.x. Using
        it emits ``FutureWarning``. It cannot be combined with
        ``transform_continuous='zscore'``.
    bin_size : int, default=120
        Number of samples per rolling window. Mørkved et al. (2015) use 120.
    zero_handles: str, default=`eps`
        Strategy to handles zeros in input data. The default behavior is to use
        Box-Cox transformations, but in case of values equal to 0, the method
        will use any of the strategy here. Could be one of `eps`, `yeojohnson`
    anchor_strategy : {'farthest_point', 'projection_rank'}, default='farthest_point'
        How the k-NN window anchors are chosen when two continuous covariates
        are used (ignored for a single covariate, which uses rolling windows).

        ``'farthest_point'`` runs greedy farthest-point sampling in the scaled
        covariate space: start from the observation closest to the centroid,
        then repeatedly take the observation whose distance to the closest
        already-chosen anchor is largest. The anchors spread over the occupied
        region, so the polynomial surface is supported across the covariate
        cloud rather than along one ridge.

        ``'projection_rank'`` is the previous behaviour: sort by the sum of the
        scaled columns and take evenly spaced rank positions. Evenly spaced
        ranks are evenly spaced in probability mass, not in covariate space, so
        on a correlated covariate pair (e.g. gestational week x birth weight,
        whose mass follows a growth curve) every anchor lands on the dense
        ridge and the surface is unidentified off it.

        Both strategies only ever return observed rows, so no anchor sits in an
        empty region. Farthest-point sampling is deterministic (no RNG) and
        costs ``O(n_samples * n_bins)``. It is, by design, attracted to extreme
        observations: a far-flung outlier becomes an anchor and its
        ``bin_size`` nearest neighbours are then largely the same bulk points a
        neighbouring anchor sees. Window coordinates stay the median of each
        window's covariates, which pulls a boundary anchor's recorded center
        back inside the cloud.
    transform_continuous : {None, 'log10', 'zscore'}, default=None
        Transformation applied to continuous covariates before binning and
        polynomial feature construction. ``None`` preserves the original
        values; ``'log10'`` applies the legacy base-10 logarithm and requires
        strictly positive values; ``'zscore'`` applies a training-set Z-score
        (subtract column mean and divide by column standard deviation). The
        fitted centering parameters are reused unchanged at prediction time.

    Attributes
    ----------
    poly_transformer : sklearn.preprocessing.PolynomialFeatures
        Transforms covariate vectors into polynomial feature matrices.
    poly_scaler : sklearn.preprocessing.StandardScaler
        Standardizes polynomial feature columns before Ridge fitting, making the
        penalty independent of the covariates' units and polynomial powers.
    mu_model : sklearn.linear_model.Ridge
        Ridge model predicting mu (in Box-Cox space) from polynomial features.
    sigma_model : sklearn.linear_model.Ridge
        Ridge model predicting log-sigma from polynomial features. Predictions
        are exponentiated back to sigma in Box-Cox space.
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
    marginal_mu_ : float or None
        Covariate-free mu (in Box-Cox space) over the whole training target.
        ``None`` until :meth:`fit` is called. Used by :meth:`predict_mu_sigma` as
        the fallback where the polynomial surface predicts an invalid scale.
    marginal_sigma_ : float or None
        Covariate-free sigma counterpart of ``marginal_mu_``, floored at ``1e-6``.
    continuous_center_ : ndarray of shape (n_features,) or None
        Training-set column means used by ``transform_continuous='zscore'``.
        ``None`` for the other transformation modes.
    continuous_scale_ : ndarray of shape (n_features,) or None
        Training-set population standard deviations used by
        ``transform_continuous='zscore'``. Near-constant columns receive scale
        1.0. ``None`` for the other transformation modes.

    Notes
    -----
    Q-Q regression estimates mu and sigma by fitting a line through the
    empirical quantiles of Box-Cox (or Jeo-Yohnson) transformed window data
    against the corresponding theoretical normal quantiles using the Blom plotting
    position formula ``(i - 3/8) / (n + 1/4)``. The slope gives sigma
    and the intercept gives mu, both in Box-Cox space. Sigma is floored
    at ``1e-6`` to prevent division by zero during Z-score computation.
    The polynomial scale surface is fitted to the logarithm of the per-window
    sigma estimates. At prediction time finite log-scale outputs are constrained
    to the range observed in the final fitting windows and exponentiated, so
    every finite scale prediction is strictly positive and remains supported by
    the fitted window estimates.
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
        anchor_strategy: str = "farthest_point",
        transform_continuous: Optional[str] = None,
        ridge_alpha: float = 0.05,
    ):
        self.n_bins = n_bins
        self.degree = degree
        self.ridge_alpha = ridge_alpha
        self.lambda_ = lambda_
        self.n_iterations = n_iterations
        self.log_transform_continuous = log_transform_continuous
        self.bin_size = bin_size
        self.poly_transformer = PolynomialFeatures(
            degree=self.degree, include_bias=True
        )
        self.poly_scaler = StandardScaler()
        self.mu_model = Ridge(alpha=self.ridge_alpha, fit_intercept=True)
        self.sigma_model = Ridge(alpha=self.ridge_alpha, fit_intercept=True)
        self._global_mu: float = 0.0
        self._global_sigma: float = 1.0
        self._is_fitted: bool = False
        self.zero_handles = zero_handles
        self.anchor_strategy = anchor_strategy
        self.transform_continuous = transform_continuous
        self.bin_centers_: Optional[np.ndarray] = None
        self.bin_mu_: Optional[np.ndarray] = None
        self.bin_sigma_: Optional[np.ndarray] = None
        self.marginal_mu_: Optional[float] = None
        self.marginal_sigma_: Optional[float] = None
        self._log_sigma_bounds_: Optional[Tuple[float, float]] = None

    def fit(self, X_cont: np.ndarray, y: np.ndarray) -> "ContinuousSurfaceFitter":
        """Fit the polynomial mu/sigma curve to rolling window estimates.

        Parameters
        ----------
        X_cont : ndarray of shape (n_samples, n_features)
            Continuous covariate matrix. Pass an array with zero columns
            (``n_features == 0``) to trigger the global fallback.
            Accepts one or two columns; two-column input uses k-NN binning.
            All values must be strictly positive when the resolved continuous
            transformation is ``'log10'``.
        y : ndarray of shape (n_samples,)
            Target values. Must be strictly positive (required by Box-Cox).

        Returns
        -------
        self : ContinuousSurfaceFitter

        Raises
        ------
        ValueError
            If ``transform_continuous`` is invalid, if the resolved logarithmic
            transformation receives a continuous covariate value <= 0, or if
            ``anchor_strategy`` is not one of ``'farthest_point'`` /
            ``'projection_rank'``.
        """
        # Checked here rather than in __init__ (sklearn convention) and for both
        # covariate counts, so a typo fails fast instead of silently on the
        # single-covariate path where the strategy is unused.
        if self.anchor_strategy not in _ANCHOR_STRATEGIES:
            raise ValueError(
                f"anchor_strategy must be one of {_ANCHOR_STRATEGIES}; "
                f"got {self.anchor_strategy!r}."
            )
        if (
            not isinstance(self.ridge_alpha, (int, float, np.number))
            or not np.isfinite(self.ridge_alpha)
            or self.ridge_alpha < 0.0
        ):
            raise ValueError(
                f"ridge_alpha must be a finite non-negative number; "
                f"got {self.ridge_alpha!r}."
            )
        _resolve_continuous_transform(
            self.transform_continuous, self.log_transform_continuous
        )
        if self.log_transform_continuous:
            _warn_log_transform_continuous_deprecated()

        _, n_features = X_cont.shape
        X_cont = self._transform_continuous_covariates(X_cont, fit=True)

        if self.lambda_ is None:
            self.lambda_ = self._find_lambda_grid_search(y)

        # Covariate-free (mu, sigma) over the whole training target, always. This is
        # the degradation target for :meth:`predict_mu_sigma` when the polynomial
        # surface turns out to be invalid at some covariate value; the ``_global_*``
        # pair cannot serve that role because it is only populated on the
        # ``_fit_fallback`` path and otherwise keeps its (0.0, 1.0) defaults.
        m_mu, m_sigma = self._robust_qq_estimation(y, self.lambda_)
        if m_mu is None or m_sigma is None:
            y_bc_full = self._transform(y, self.lambda_)
            m_mu, m_sigma = float(np.mean(y_bc_full)), float(np.std(y_bc_full))
        self.marginal_mu_ = float(m_mu)
        self.marginal_sigma_ = max(float(m_sigma), _SIGMA_FLOOR)

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
            X_poly_scaled = self.poly_scaler.fit_transform(X_poly)
            effective_alpha = float(self.ridge_alpha) * len(valid_centers)
            self.mu_model.set_params(alpha=effective_alpha)
            self.sigma_model.set_params(alpha=effective_alpha)
            self.mu_model.fit(X_poly_scaled, mu_estimates)
            log_sigma_estimates = np.log(
                np.maximum(np.asarray(sigma_estimates), _SIGMA_FLOOR)
            )
            self.sigma_model.fit(X_poly_scaled, log_sigma_estimates)
            self._log_sigma_bounds_ = (
                float(np.min(log_sigma_estimates)),
                float(np.max(log_sigma_estimates)),
            )
            self._is_fitted = True
            self.bin_centers_ = np.array(valid_centers)
            self.bin_mu_ = np.array(mu_estimates)
            self.bin_sigma_ = np.array(sigma_estimates)

            X_poly_work = self.poly_transformer.transform(X_work)
            X_poly_work_scaled = self.poly_scaler.transform(X_poly_work)
            mu_pred = self.mu_model.predict(X_poly_work_scaled)
            sigma_pred = self._predict_sigma_from_poly(X_poly_work_scaled)
            y_bc = self._transform(y_work, self.lambda_)

            z = (y_bc - mu_pred) / sigma_pred
            mask = np.abs(z) <= _ZSCORE_OUTLIER_THRESHOLD

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
            columns as the array passed to :meth:`fit`. All values must be
            strictly positive when the fitted transformation is ``'log10'``.

        Returns
        -------
        mu : ndarray of shape (n_samples,)
            Predicted location parameter (in Box-Cox space) for each sample.
        sigma : ndarray of shape (n_samples,)
            Predicted scale parameter (in Box-Cox space). Always positive.

        Warns
        -----
        UserWarning
            When the polynomial predicts a non-finite location or log-scale for
            one or more samples. Such a sample sits where the surface is not
            supported by the training covariates. Both ``mu`` and ``sigma``
            degrade to the marginal fit there.
        """
        if not self._is_fitted or X_cont.shape[1] == 0:
            return (
                np.full(X_cont.shape[0], self._global_mu),
                np.full(X_cont.shape[0], self._global_sigma),
            )
        X_cont = self._transform_continuous_covariates(X_cont, fit=False)
        X_poly = self.poly_transformer.transform(X_cont)
        X_poly_scaled = self.poly_scaler.transform(X_poly)
        mu = self.mu_model.predict(X_poly_scaled)
        sigma = self._predict_sigma_from_poly(X_poly_scaled)

        invalid = ~np.isfinite(sigma) | (sigma <= 0.0) | ~np.isfinite(mu)
        if invalid.any():
            warnings.warn(
                f"Polynomial surface predicted an invalid mu or sigma for "
                f"{int(invalid.sum())} of {invalid.size} samples; those covariate "
                f"values are outside the region the surface supports. Falling back "
                f"to the marginal (covariate-free) fit for them."
            )
            mu = np.where(invalid, self.marginal_mu_, mu)
            sigma = np.where(invalid, self.marginal_sigma_, sigma)
        return mu, sigma

    def _predict_sigma_from_poly(self, X_poly: np.ndarray) -> np.ndarray:
        """Predict a strictly positive sigma from polynomial features.

        ``sigma_model`` owns the unconstrained log-scale surface. Finite outputs
        are constrained to the range of log-scales actually estimated in the
        final fitting windows, preventing unsupported polynomial oscillations
        from becoming near-zero or enormous scales after exponentiation. The
        lower bound is never below ``log(1e-6)``. Non-finite predictions remain
        non-finite so :meth:`predict_mu_sigma` can degrade them to the marginal
        fit instead of silently treating them as valid scales.
        """
        log_sigma = self.sigma_model.predict(X_poly)
        lower, upper = self._log_sigma_bounds_
        lower = max(lower, _LOG_SIGMA_FLOOR)
        finite = np.isfinite(log_sigma)
        bounded_log_sigma = log_sigma.copy()
        bounded_log_sigma[finite] = np.clip(log_sigma[finite], lower, upper)
        with np.errstate(over="ignore", invalid="ignore"):
            return np.exp(bounded_log_sigma)

    def _transform_continuous_covariates(
        self, X_cont: np.ndarray, *, fit: bool
    ) -> np.ndarray:
        """Apply the configured continuous-covariate transformation.

        ``'zscore'`` subtracts the training mean and divides by the training
        population standard deviation. The fitted parameters are stored so
        predictions never depend on the composition of the prediction batch.
        """
        if fit:
            resolved = _resolve_continuous_transform(
                self.transform_continuous, self.log_transform_continuous
            )
            self._resolved_transform_continuous_ = resolved
            self.continuous_center_ = None
            self.continuous_scale_ = None
        else:
            resolved = self._resolved_transform_continuous_

        if resolved == "log10":
            for c in range(X_cont.shape[1]):
                if np.any(X_cont[:, c] <= 0):
                    raise ValueError(
                        "transform_continuous='log10' requires strictly positive "
                        f"covariate values; column {c} contains values <= 0."
                    )
            return np.log10(X_cont)

        if resolved == "zscore":
            if fit:
                self.continuous_center_ = np.mean(X_cont, axis=0)
                scale = np.std(X_cont, axis=0)
                self.continuous_scale_ = np.where(scale < 1e-8, 1.0, scale)
            return (X_cont - self.continuous_center_) / self.continuous_scale_

        return X_cont

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

    def _select_reference_points(self, X_scaled: np.ndarray) -> np.ndarray:
        """Pick the k-NN window anchors among the observed rows.

        Parameters
        ----------
        X_scaled : ndarray of shape (n_samples, n_features)
            Covariate matrix centred and scaled per column. Must have at least
            one row.

        Returns
        -------
        ref_idx : ndarray of shape (n_anchors,)
            Sorted, unique row indices into ``X_scaled``. At most ``n_bins``
            anchors, fewer when ``n_samples < n_bins`` or when a strategy picks
            the same row twice.

        Raises
        ------
        ValueError
            If ``anchor_strategy`` is unknown. :meth:`fit` already rejects that,
            so this only fires when the attribute is set after construction.
        """
        n = X_scaled.shape[0]

        if self.anchor_strategy == "farthest_point":
            # Greedy farthest-point sampling: seed at the most central
            # observation, then repeatedly take the row whose distance to the
            # nearest already-chosen anchor is largest. argmin/argmax return the
            # first extremum, so the result is deterministic.
            centroid = X_scaled.mean(axis=0)
            chosen = [int(np.argmin(np.linalg.norm(X_scaled - centroid, axis=1)))]
            dmin = np.linalg.norm(X_scaled - X_scaled[chosen[0]], axis=1)
            # Capped at n: once every row is an anchor dmin is all-zero and
            # further picks are duplicates that np.unique collapses anyway.
            for _ in range(min(self.n_bins, n) - 1):
                nxt = int(np.argmax(dmin))
                chosen.append(nxt)
                dmin = np.minimum(
                    dmin, np.linalg.norm(X_scaled - X_scaled[nxt], axis=1)
                )
            return np.unique(chosen)

        if self.anchor_strategy == "projection_rank":
            proj_sorted_idx = np.argsort(X_scaled.sum(axis=1))
            ref_positions = np.round(np.linspace(0, n - 1, self.n_bins)).astype(int)
            return np.unique(proj_sorted_idx[ref_positions])

        raise ValueError(
            f"anchor_strategy must be one of {_ANCHOR_STRATEGIES}; "
            f"got {self.anchor_strategy!r}."
        )

    def _create_knn_bins(self, X: np.ndarray, y: np.ndarray) -> Tuple[List, List, List]:
        """Build k-NN overlapping windows for a 2D continuous covariate space.

        Selects up to ``n_bins`` reference points among the observed rows via
        ``anchor_strategy`` (see the class docstring), then for each reference
        point the ``bin_size`` nearest neighbours in the scaled 2D space form a
        bin.  This avoids grid-based binning and the curse of dimensionality.

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

        Notes
        -----
        Returns three empty lists when ``y`` has zero rows, instead of
        raising, so callers can degrade gracefully on an empty input.
        """
        n = len(y)
        if n == 0:
            return [], [], []
        x_std = X.std(axis=0)
        x_std[x_std < 1e-8] = 1.0
        X_scaled = (X - X.mean(axis=0)) / x_std

        ref_idx = self._select_reference_points(X_scaled)

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
        return mu, max(sigma, _SIGMA_FLOOR)

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
            self._global_sigma = max(float(np.std(y_bc)), _SIGMA_FLOOR)
        else:
            self._global_mu = mu
            self._global_sigma = sigma
        self._log_sigma_bounds_ = None
        self._is_fitted = False


__all__ = [
    "RobustNormalizerConfig",
    "ContinuousSurfaceFitter",
]
