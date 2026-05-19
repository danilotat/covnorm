import warnings
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from scipy.stats import boxcox, yeojohnson
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import validate_data

from covnorm._surface_fitter import ContinuousSurfaceFitter, RobustNormalizerConfig


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
    zero_handles: str, default=`eps`
        Strategy to handles zeros in input data. The default behavior is to use
        Box-Cox transformations, but in case of values equal to 0, the method
        will use any of the strategy here. Could be one of `eps`, `yeojohnson`

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
        zero_handles: str = "eps",
    ):
        self.categorical_cols = categorical_cols
        self.continuous_cols = continuous_cols
        self.target_col = target_col
        self.n_bins = n_bins
        self.degree = degree
        self.n_iterations = n_iterations
        self.log_transform_continuous = log_transform_continuous
        self.bin_size = bin_size
        self.zero_handles = zero_handles
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
        validate_data(self, X, reset=True)
        self._resolved_target_cols = self._resolve_target_cols(X.shape[1])
        if self.zero_handles.lower() not in ('eps', 'yeojohnson'):
            raise ValueError("zero_handles must be 'eps' or 'yeojohnson'.")

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
                zero_handles=self.zero_handles,
            )
            fitter.fit(X_cont_all, y_all)
            self._fitters[col] = fitter
            if np.any(y_all == 0):
                if self.zero_handles.lower() == 'eps':
                    y_all = y_all + 1e-6
                    y_bc_all = boxcox(y_all, lmbda=fitter.lambda_)
                elif self.zero_handles.lower() == 'yeojohnson':
                    y_bc_all = yeojohnson(y_all, lmbda=fitter.lambda_)
            else:
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
        validate_data(self, X_out, reset=False)
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

            if np.any(y_raw < 0) or (self.zero_handles.lower() == 'eps' and np.any(y_raw == 0)):
                raise ValueError(
                    f"Column {col} contains non-positive values; Box-Cox requires "
                    "strictly positive data. Use zero_handles='yeojohnson' if zeros are present."
                )

            fitter = self._fitters[col]
            if np.any(y_raw==0):
                if self.zero_handles.lower() == 'eps':
                    y_raw = y_raw + 1e-6
                    y_bc = boxcox(y_raw, lmbda=fitter.lambda_)
                elif self.zero_handles.lower() == 'yeojohnson':
                    y_bc = yeojohnson(y_raw, lmbda=fitter.lambda_)
            else:
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
