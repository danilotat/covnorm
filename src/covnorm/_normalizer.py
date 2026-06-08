import warnings
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from numpy.typing import ArrayLike
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
    categorical_vals : ArrayLike of shape (n_samples, n_cat) or (n_samples,)
        Array of categorical covariate values used to compute post-hoc
        group corrections (e.g. sex, batch). At most
        ``RobustNormalizerConfig.MAX_CATEGORICAL`` (default 2) columns.
        Pass an empty list ``[]`` when there are no categorical covariates.
    continuous_vals : ArrayLike of shape (n_samples, n_cont) or (n_samples,)
        Array of continuous covariate values used to model within-group
        variation of mu and sigma (e.g. age, BMI). Up to two columns are
        supported; one column uses 1D rolling windows, two columns use k-NN
        windows (``RobustNormalizerConfig.MAX_CONTINUOUS == 2``).
        When two columns are used, set ``n_bins >= 10`` so that the number
        of valid centers meets the minimum for a degree-3 surface polynomial
        (``comb(3+2, 2) == 10`` terms).
        Pass an empty list ``[]`` when there are no continuous covariates.
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
    zero_handles : str, default='eps'
        Strategy to handle zeros in input data. The default behavior is to use
        Box-Cox transformations, but in case of values equal to 0, the method
        will use any of the strategy here. Could be one of ``'eps'``,
        ``'yeojohnson'``.

    Attributes
    ----------
    _fitters : dict of int -> ContinuousSurfaceFitter
        One polynomial surface per marker column, populated by :meth:`fit`.
    _cat_corrections : dict of int -> (dict of tuple -> (float, float))
        Per-marker-column mapping from categorical combination to
        ``(mu_cat, sigma_cat)`` correction parameters.

    Notes
    -----
    ``X`` passed to :meth:`fit` and :meth:`transform` must be a 2-D array
    containing **only the marker/target columns** — covariate data is provided
    separately via ``categorical_vals`` and ``continuous_vals`` at
    construction time (or overridden at :meth:`transform` time).

    Unseen categorical combinations at transform time produce a warning and
    are assigned a Z-score of 0.

    Examples
    --------
    >>> import numpy as np
    >>> from covnorm import RobustConditionalNormalizer
    >>> rng = np.random.default_rng(0)
    >>> sex = rng.integers(0, 2, 500).astype(float).reshape(-1, 1)
    >>> age = rng.uniform(1, 80, (500, 1))
    >>> marker = rng.gamma(2, 10, (500, 1))
    >>> norm = RobustConditionalNormalizer(
    ...     categorical_vals=sex,
    ...     continuous_vals=age,
    ...     log_transform_continuous=True,
    ... )
    >>> marker_norm = norm.fit_transform(marker)
    >>> marker_norm.shape
    (500, 1)
    """

    def __init__(
        self,
        categorical_vals: ArrayLike,
        continuous_vals: ArrayLike,
        n_bins: int = 6,
        degree: int = 3,
        n_iterations: int = 3,
        log_transform_continuous: bool = False,
        bin_size: int = 120,
        zero_handles: str = "eps",
    ):
        self.categorical_vals = categorical_vals
        self.continuous_vals = continuous_vals
        self.n_bins = n_bins
        self.degree = degree
        self.n_iterations = n_iterations
        self.log_transform_continuous = log_transform_continuous
        self.bin_size = bin_size
        self.zero_handles = zero_handles
        self._fitters: Dict[int, ContinuousSurfaceFitter] = {}
        self._cat_corrections: Dict[int, Dict[Tuple, Tuple[float, float]]] = {}
        self._cat_encoders: Dict[int, Dict] = {}
        self._validate_constraints()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _n_covariate_cols(vals: ArrayLike) -> int:
        """Return the number of covariate columns in an ArrayLike."""
        arr = np.asarray(vals)
        if arr.size == 0:
            return 0
        return int(arr.shape[1]) if arr.ndim >= 2 else 1

    @staticmethod
    def _coerce_covariates(vals: ArrayLike, n_samples: int) -> np.ndarray:
        """Return a (n_samples, n_cols) float array; empty vals → (n_samples, 0)."""
        arr = np.asarray(vals)
        if arr.size == 0:
            return np.empty((n_samples, 0), dtype=float)
        arr = arr.astype(float)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        return arr

    def _encode_categoricals(self, cat: np.ndarray, fit: bool = False) -> np.ndarray:
        """Encode categorical covariate array to float, handling string labels."""
        if cat.dtype.kind in "iufcb":
            return cat.astype(float)
        if fit:
            self._cat_encoders = {
                c: {v: float(i) for i, v in enumerate(sorted(set(cat[:, c]), key=str))}
                for c in range(cat.shape[1])
            }
        out = np.empty(cat.shape, dtype=float)
        for j in range(cat.shape[1]):
            enc = self._cat_encoders.get(j)
            out[:, j] = [enc[v] for v in cat[:, j]] if enc else cat[:, j].astype(float)
        return out

    def _validate_constraints(self) -> None:
        n_cat = self._n_covariate_cols(self.categorical_vals)
        if n_cat > RobustNormalizerConfig.MAX_CATEGORICAL:
            raise ValueError(
                f"Exceeded max categorical covariates ({RobustNormalizerConfig.MAX_CATEGORICAL})."
            )
        n_cont = self._n_covariate_cols(self.continuous_vals)
        if n_cont > RobustNormalizerConfig.MAX_CONTINUOUS:
            raise ValueError(
                f"Exceeded max continuous covariates ({RobustNormalizerConfig.MAX_CONTINUOUS})."
            )

    def fit(
        self, X: np.ndarray, y: Optional[np.ndarray] = None
    ) -> "RobustConditionalNormalizer":
        """Fit a single combined surface and compute per-group categorical corrections.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_markers) or (n_samples,)
            Marker/target matrix. Every column is treated as a target to
            normalize. Values must be strictly positive (required by Box-Cox)
            unless ``zero_handles='yeojohnson'`` is set.
        y : ignored
            Not used. Present for scikit-learn pipeline compatibility.

        Returns
        -------
        self : RobustConditionalNormalizer
            Fitted estimator.
        """
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        validate_data(self, X, reset=True)
        n_samples = X.shape[0]

        if self.zero_handles.lower() not in ("eps", "yeojohnson"):
            raise ValueError("zero_handles must be 'eps' or 'yeojohnson'.")

        cat_raw = self._coerce_covariates(self.categorical_vals, n_samples)
        cont_data = self._coerce_covariates(self.continuous_vals, n_samples)

        if cat_raw.shape[0] != n_samples:
            raise ValueError(
                f"categorical_vals has {cat_raw.shape[0]} rows but X has {n_samples}."
            )
        if cont_data.shape[0] != n_samples:
            raise ValueError(
                f"continuous_vals has {cont_data.shape[0]} rows but X has {n_samples}."
            )

        if cat_raw.shape[1] > 0:
            cat_data = self._encode_categoricals(cat_raw, fit=True)
            unique_rows, inverse_indices = np.unique(
                cat_data, axis=0, return_inverse=True
            )
        else:
            cat_data = cat_raw
            unique_rows = inverse_indices = None

        self._fitters = {}
        self._cat_corrections = {}

        for col in range(X.shape[1]):
            y_all = X[:, col]

            fitter = ContinuousSurfaceFitter(
                n_bins=self.n_bins,
                degree=self.degree,
                n_iterations=self.n_iterations,
                log_transform_continuous=self.log_transform_continuous,
                bin_size=self.bin_size,
                zero_handles=self.zero_handles,
            )
            fitter.fit(cont_data, y_all)
            self._fitters[col] = fitter
            y_bc_all = fitter._transform(y_all, fitter.lambda_)

            mu_all, sigma_all = fitter.predict_mu_sigma(cont_data)
            z_base_all = (y_bc_all - mu_all) / np.maximum(sigma_all, 1e-6)

            col_corrections: Dict[Tuple, Tuple[float, float]] = {}
            if cat_data.shape[1] == 0:
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

    def transform(
        self,
        X: np.ndarray,
        categorical_vals: Optional[ArrayLike] = None,
        continuous_vals: Optional[ArrayLike] = None,
    ) -> np.ndarray:
        """Apply conditional Z-score normalization with post-hoc categorical correction.

        Each sample's base Z-score is computed as
        ``(BoxCox(y) - mu(x)) / sigma(x)`` using the shared polynomial surface,
        then corrected for its categorical group:
        ``Z_corrected = (Z_base - mu_cat) / sigma_cat``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_markers) or (n_samples,)
            Marker matrix with the same number of columns as used in
            :meth:`fit`. Values must be strictly positive (required by
            Box-Cox) unless ``zero_handles='yeojohnson'`` is set.
        categorical_vals : ArrayLike, optional
            Override the categorical covariate values stored at construction.
            Must have the same number of columns as the training
            ``categorical_vals``. Useful for applying a fitted normalizer to
            new samples with different categorical covariate values.
        continuous_vals : ArrayLike, optional
            Override the continuous covariate values stored at construction.

        Returns
        -------
        X_out : ndarray of shape (n_samples, n_markers)
            Z-scored marker values. Shape matches the input ``X``; no
            covariate columns are included in the output.

        Warns
        -----
        UserWarning
            Raised for any categorical combination not seen during
            :meth:`fit`. Affected samples receive a Z-score of 0.
        """
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        validate_data(self, X, reset=False)
        n_samples = X.shape[0]

        cat_src = (
            categorical_vals if categorical_vals is not None else self.categorical_vals
        )
        cont_src = (
            continuous_vals if continuous_vals is not None else self.continuous_vals
        )

        cat_raw = self._coerce_covariates(cat_src, n_samples)
        cont_data = self._coerce_covariates(cont_src, n_samples)

        if cat_raw.shape[1] > 0:
            cat_data = self._encode_categoricals(cat_raw, fit=False)
            unique_rows, inverse_indices = np.unique(
                cat_data, axis=0, return_inverse=True
            )
            cat_groups = [
                (tuple(unique_rows[i]), np.where(inverse_indices == i)[0])
                for i in range(len(unique_rows))
            ]
        else:
            cat_groups = [(tuple(), np.arange(n_samples))]

        X_out = X.copy()

        for col in range(X.shape[1]):
            y_raw = X_out[:, col]
            fitter = self._fitters[col]
            y_bc = fitter._transform(y_raw, fitter.lambda_)

            mu_pred, sigma_pred = fitter.predict_mu_sigma(cont_data)
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
        marker = rng.gamma(shape=2, scale=10, size=n_per_group)
        cat = np.full(n_per_group, float(group_id))
        groups.append(np.column_stack([cat, age, marker]))
    data = np.vstack(groups)

    cat_col = data[:, [0]]
    cont_col = data[:, [1]]
    markers = data[:, [2]]

    norm = RobustConditionalNormalizer(
        categorical_vals=cat_col,
        continuous_vals=cont_col,
        log_transform_continuous=True,
    )
    markers_norm = norm.fit_transform(markers)

    assert (
        markers_norm.shape == markers.shape
    ), f"Shape mismatch: {markers_norm.shape} != {markers.shape}"

    for group_id in (0, 1):
        mask = data[:, 0] == float(group_id)
        z = markers_norm[mask, 0]
        mean_z = float(np.mean(z))
        std_z = float(np.std(z))
        assert (
            abs(mean_z) < 0.1
        ), f"Group {group_id}: mean z = {mean_z:.4f}, expected |mean| < 0.1"
        assert (
            abs(std_z - 1.0) < 0.2
        ), f"Group {group_id}: std z = {std_z:.4f}, expected |std - 1| < 0.2"

    print("PASS")
