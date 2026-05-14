# covnorm

Scikit-learn compatible transformer for robust conditional Z-score normalization of a continuous marker against categorical and continuous covariates.

For each unique combination of categorical covariates, the transformer fits a polynomial surface over mu and sigma estimated per bin via Q-Q regression with Tukey's fence outlier rejection. At transform time, each sample is normalized using the mu and sigma predicted from its covariate values.

Supports up to 2 categorical and 2 continuous covariates.

## Installation

```bash
pip install covnorm
```

## Usage

```python
import numpy as np
from covnorm import RobustConditionalNormalizer

# X columns: [sex, batch, age, cell_count, marker]
# sex=0, batch=1 → categorical
# age=2, cell_count=3 → continuous
# marker=4 → target to normalize

X = np.load("data.npy")  # shape (n_samples, 5)

normalizer = RobustConditionalNormalizer(
    categorical_cols=[0, 1],
    continuous_cols=[2, 3],
    target_col=4,
    n_bins=6,   # bins per continuous covariate
    degree=2,   # polynomial degree for surface fitting
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
| `n_bins` | `6` | Number of bins per continuous covariate axis |
| `degree` | `2` | Polynomial degree of the mu/sigma surface |
