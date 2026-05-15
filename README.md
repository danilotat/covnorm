# covnorm

Scikit-learn compatible transformer for robust conditional Z-score normalization of a continuous marker against categorical and continuous covariates.

A single polynomial curve is fitted over mu and sigma across the entire dataset using rolling overlapping bins sorted by the continuous covariate. Mu and sigma per bin are estimated via Q-Q regression on Box-Cox transformed values, with iterative conditional Z-score outlier rejection (samples with `|Z| > 3.372` against the fitted surface are dropped and the surface is refit, repeated up to `n_iterations` times). The Box-Cox lambda is selected by a grid search that maximises Q-Q linearity rather than marginal normality.

After fitting the shared surface, a post-hoc categorical correction is applied: the mean and standard deviation of the base Z-scores are computed within each categorical group and used to rescale final Z-scores, giving `Z_corrected = (Z_base - mu_cat) / sigma_cat`. This avoids overfitting independent surfaces to small groups.

> **Data requirement:** target values must be strictly positive (Box-Cox transform). Shift your data if it contains zeros or negatives.

Supports up to 2 categorical covariates and exactly 1 continuous covariate.

## Installation

```bash
pip install covnorm
```

## Usage

```python
import numpy as np
from covnorm import RobustConditionalNormalizer

# X columns: [sex, batch, age, marker]
# sex=0, batch=1 → categorical
# age=2 → continuous (exactly one supported)
# marker=3 → target to normalize

X = np.load("data.npy")  # shape (n_samples, 4)

normalizer = RobustConditionalNormalizer(
    categorical_cols=[0, 1],
    continuous_cols=[2],
    target_col=3,
    n_bins=6,                      # target number of rolling windows
    bin_size=120,                  # samples per rolling window
    log_transform_continuous=True, # recommended when covariates span orders of magnitude
)

X_norm = normalizer.fit_transform(X)
```

The target column in `X_norm` contains Z-scores. All other columns are unchanged.

It follows the scikit-learn `fit` / `transform` / `fit_transform` API and is compatible with `Pipeline`.

## Parameters

| Parameter | Default | Description |
|---|---|---|
| `categorical_cols` | — | Column indices treated as categorical grouping variables |
| `continuous_cols` | — | Column indices used as continuous covariates for surface fitting |
| `target_col` | — | Column index of the marker to normalize |
| `n_bins` | `6` | Target number of rolling windows (controls stride) |
| `bin_size` | `120` | Samples per rolling window (Mørkved et al. use 120) |
| `degree` | `3` | Polynomial degree of the mu/sigma curve |
| `n_iterations` | `3` | Maximum iterative conditional outlier-removal passes |
| `log_transform_continuous` | `False` | Apply log10 to the continuous covariate before fitting (recommended when it spans orders of magnitude, e.g. age in years) |
