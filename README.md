# covnorm

Scikit-learn compatible transformer for robust conditional Z-score normalization of a continuous marker against categorical and continuous covariates.

A single polynomial curve is fitted over mu and sigma across the entire dataset using rolling overlapping bins sorted by the continuous covariate. Mu and sigma per bin are estimated via Q-Q regression on Box-Cox transformed values, with iterative conditional Z-score outlier rejection (samples with `|Z| > 3.372` against the fitted surface are dropped and the surface is refit, repeated up to `n_iterations` times). The Box-Cox lambda is selected by a grid search that maximises Q-Q linearity rather than marginal normality.

After fitting the shared surface, a post-hoc categorical correction is applied: the mean and standard deviation of the base Z-scores are computed within each categorical group and used to rescale final Z-scores, giving `Z_corrected = (Z_base - mu_cat) / sigma_cat`. This avoids overfitting independent surfaces to small groups.

> **Data requirement:** target values must be strictly positive for Box-Cox (default). If your data contains zeros, set `zero_handles="eps"` (adds a small epsilon before Box-Cox) or `zero_handles="yeojohnson"` (switches to Yeo-Johnson transform). Negative values are not supported.

Supports up to 2 categorical covariates and up to 2 continuous covariates. With 2 continuous covariates, k-NN overlapping windows replace the 1D rolling windows; set `n_bins >= 10` in that case.

## Installation

```bash
pip install covnorm
```

## Usage

```python
import numpy as np
from covnorm import RobustConditionalNormalizer

# Separate covariates from markers before passing to the normalizer.
data = np.load("data.npy")  # shape (n_samples, 4): [sex, batch, age, marker]

sex_batch = data[:, [0, 1]]  # categorical covariates, shape (n_samples, 2)
age       = data[:, [2]]     # continuous covariate,  shape (n_samples, 1)
marker    = data[:, [3]]     # marker to normalize,   shape (n_samples, 1)

normalizer = RobustConditionalNormalizer(
    categorical_vals=sex_batch,
    continuous_vals=age,
    n_bins=6,                      # target number of rolling windows
    bin_size=120,                  # samples per rolling window
    transform_continuous="zscore", # training-set Z-score for polynomial stability
)

marker_norm = normalizer.fit_transform(marker)
```

`marker_norm` has shape `(n_samples, n_markers)` and contains only the Z-scored marker columns — no covariate columns are included in the output.

It follows the scikit-learn `fit` / `transform` / `fit_transform` API and is compatible with `Pipeline`.

### Inference on new samples

`transform()` accepts optional `categorical_vals` and `continuous_vals` overrides so a fitted normalizer can be applied to new samples with different covariate values:

```python
marker_new_norm = normalizer.transform(
    marker_new,
    categorical_vals=sex_batch_new,
    continuous_vals=age_new,
)
```

> **Important:** when `marker_new` contains fewer rows than the training set (including the common case of a single sample), you **must** pass `categorical_vals` and `continuous_vals` whose row count matches `marker_new.shape[0]`. If the overrides are omitted, `transform()` falls back to the training-time covariate arrays, which have a different number of rows, causing an `IndexError`.

## Parameters

| Parameter | Default | Description |
|---|---|---|
| `categorical_vals` | — | Array of shape `(n_samples, n_cat)` or `(n_samples,)` with categorical covariate values (e.g. sex, batch). Pass `[]` when there are no categorical covariates. |
| `continuous_vals` | — | Array of shape `(n_samples, n_cont)` or `(n_samples,)` with continuous covariate values (e.g. age, BMI). Pass `[]` when there are no continuous covariates. |
| `n_bins` | `6` | Target number of rolling windows (controls stride) |
| `bin_size` | `120` | Samples per rolling window (Mørkved et al. use 120) |
| `degree` | `3` | Polynomial degree of the mu/sigma curve |
| `ridge_alpha` | `0.05` | L2 penalty relative to mean squared fitting loss. Internally multiplied by the number of valid bins; set to `0.0` to disable regularization. |
| `n_iterations` | `3` | Maximum iterative conditional outlier-removal passes |
| `transform_continuous` | `None` | Continuous-covariate transform: `None` keeps the original values, `"log10"` applies a base-10 logarithm, and `"zscore"` subtracts the training mean and divides by the population standard deviation. Fitted parameters are reused at inference. |
| `log_transform_continuous` | `False` | Deprecated alias for `transform_continuous="log10"`. Emits `FutureWarning`, cannot be combined with `"zscore"`, and will be removed in version 1.x. |
| `zero_handles` | `"eps"` | Strategy for zero values in the target: `"eps"` adds a small epsilon before Box-Cox; `"yeojohnson"` switches to Yeo-Johnson transform (supports zeros and negatives) |

`transform_continuous` acts on the continuous covariates, whereas Box-Cox or
Yeo-Johnson acts on each marker. The two transformations serve different
purposes. `"zscore"` makes the covariate coordinates invariant to affine unit
changes without changing the polynomial function class. `"log10"` is a
non-linear modelling choice and requires all continuous covariates to be
strictly positive.

The conditional location surface is fitted directly in transformed-marker
space. The conditional scale surface is fitted to the logarithm of the
per-window sigma estimates and exponentiated at prediction time. This keeps
every finite sigma prediction strictly positive. Before exponentiation, the
predicted log-scale is constrained to the range estimated in the final fitting
windows (whose lower bound is itself floored at `log(1e-6)`). This prevents
unsupported polynomial oscillations from becoming near-zero or enormous
scales.

Before fitting, polynomial feature columns are standardized and both the
location and log-scale surfaces use Ridge regression. With `B` valid bins, the
effective objective is `mean_squared_error + ridge_alpha * ||coef||²`; the
intercept is not penalized.

## Worm-plot diagnostics

```python
from covnorm import plot_worm

fig = plot_worm(
    normalizer,
    marker,
    n_bins=6,
    covariate_label="age",
    marker_label="marker",
)
fig.savefig("marker_worm_plot.png", dpi=150)
```

`plot_worm` draws detrended normal Q-Q plots of the final normalized values.
With continuous covariates, observations are split into equal-count,
non-overlapping bins of the selected covariate. A well-calibrated normalizer
produces approximately flat worms around zero inside the pointwise reference
bands.

Offsets indicate conditional location bias, slopes indicate scale bias, and
curved patterns can reveal residual skewness or tail-weight mismatch. When two
continuous covariates are present, use `covariate_index=0` or `1` to choose the
conditioning variable. For new or held-out samples, pass matching
`categorical_vals=` and `continuous_vals=` just as for `transform()`; held-out
plots are preferable because their reference bands are not adjusted for fitting
the normalizer on the same observations.
