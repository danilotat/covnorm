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
    log_transform_continuous=True, # recommended when covariates span orders of magnitude
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

## Parameters

| Parameter | Default | Description |
|---|---|---|
| `categorical_vals` | — | Array of shape `(n_samples, n_cat)` or `(n_samples,)` with categorical covariate values (e.g. sex, batch). Pass `[]` when there are no categorical covariates. |
| `continuous_vals` | — | Array of shape `(n_samples, n_cont)` or `(n_samples,)` with continuous covariate values (e.g. age, BMI). Pass `[]` when there are no continuous covariates. |
| `n_bins` | `6` | Target number of rolling windows (controls stride) |
| `bin_size` | `120` | Samples per rolling window (Mørkved et al. use 120) |
| `degree` | `3` | Polynomial degree of the mu/sigma curve |
| `n_iterations` | `3` | Maximum iterative conditional outlier-removal passes |
| `log_transform_continuous` | `False` | Apply log10 to the continuous covariate before fitting (recommended when it spans orders of magnitude, e.g. age in years) |
| `zero_handles` | `"eps"` | Strategy for zero values in the target: `"eps"` adds a small epsilon before Box-Cox; `"yeojohnson"` switches to Yeo-Johnson transform (supports zeros and negatives) |

## Plotting

```python
from covnorm import plot_covariate_space

fig = plot_covariate_space(
    normalizer,
    data,
    covariate_labels=["age"],
    analyte_label="marker",
)
fig.savefig("normalization_surface.png", dpi=150)
```

`plot_covariate_space` visualises the fitted polynomial surface on top of the raw data:

- **1 continuous covariate** — scatter of raw values with a filled ribbon `μ(x) ± n_sigma · σ(x)` back-transformed to the original space.
- **2 continuous covariates** — 3-D surface plot of `μ(x₁, x₂)` with the per-bin estimates scattered on top.
