"""
Tests for RobustConditionalNormalizer and ContinuousSurfaceFitter.

Data layout used throughout: [sex, batch, age, marker]
  col 0 - sex        (categorical, values 0/1)
  col 1 - batch      (categorical, values 0/1/2)
  col 2 - age        (continuous)
  col 3 - marker     (target)
"""

import warnings

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.utils.estimator_checks import parametrize_with_checks

from covnorm import ContinuousSurfaceFitter, RobustConditionalNormalizer

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(42)


def _make_data(n_per_group: int = 300, rng=None) -> np.ndarray:
    """
    Two-group dataset: sex in {0, 1}.
    Each group has a distinct (mu, sigma) for the marker so normalization
    correctness is verifiable. Age is continuous and independent of the marker.

    Returns X of shape (2*n_per_group, 4): [sex, batch, age, marker]
    """
    if rng is None:
        rng = RNG

    # Gamma distribution: right-skewed, strictly positive — required by Box-Cox.
    # Groups differ in scale so between-group normalization correctness is verifiable.
    groups = [
        dict(sex=0, batch=0, shape=4.0, scale=2.0),
        dict(sex=1, batch=1, shape=4.0, scale=10.0),
    ]
    blocks = []
    for g in groups:
        age = rng.uniform(20, 80, n_per_group)
        marker = rng.gamma(g["shape"], g["scale"], n_per_group)
        sex = np.full(n_per_group, g["sex"])
        batch = np.full(n_per_group, g["batch"])
        blocks.append(np.column_stack([sex, batch, age, marker]))

    return np.vstack(blocks)


@pytest.fixture(scope="module")
def data() -> np.ndarray:
    return _make_data()


@pytest.fixture(scope="module")
def normalizer_full(data) -> RobustConditionalNormalizer:
    norm = RobustConditionalNormalizer(
        categorical_cols=[0, 1],
        continuous_cols=[2],
        target_col=3,
    )
    norm.fit(data)
    return norm


# ---------------------------------------------------------------------------
# sklearn estimator contract
# ---------------------------------------------------------------------------


def test_get_params():
    norm = RobustConditionalNormalizer(
        categorical_cols=[0], continuous_cols=[1], target_col=2, n_bins=8, degree=3
    )
    params = norm.get_params()
    assert params["categorical_cols"] == [0]
    assert params["continuous_cols"] == [1]
    assert params["target_col"] == 2
    assert params["n_bins"] == 8
    assert params["degree"] == 3


def test_set_params():
    norm = RobustConditionalNormalizer(
        categorical_cols=[0], continuous_cols=[1], target_col=2
    )
    norm.set_params(n_bins=10, degree=1)
    assert norm.n_bins == 10
    assert norm.degree == 1


def test_clone_produces_unfitted_copy(normalizer_full, data):
    cloned = clone(normalizer_full)
    assert cloned._cat_corrections == {}
    # cloned estimator must be independently fittable
    X_norm = cloned.fit_transform(data)
    assert X_norm.shape == data.shape


def test_pipeline_compatible(data):
    pipe = Pipeline(
        [
            (
                "norm",
                RobustConditionalNormalizer(
                    categorical_cols=[0, 1], continuous_cols=[2], target_col=3
                ),
            )
        ]
    )
    X_norm = pipe.fit_transform(data)
    assert X_norm.shape == data.shape


# ---------------------------------------------------------------------------
# Output contract
# ---------------------------------------------------------------------------


def test_output_shape(normalizer_full, data):
    X_norm = normalizer_full.transform(data)
    assert X_norm.shape == data.shape


def test_nontarget_columns_unchanged(normalizer_full, data):
    X_norm = normalizer_full.transform(data)
    for col in [0, 1, 2]:
        np.testing.assert_array_equal(X_norm[:, col], data[:, col])


def test_input_array_not_mutated(data):
    X_copy = data.copy()
    norm = RobustConditionalNormalizer(
        categorical_cols=[0, 1], continuous_cols=[2], target_col=3
    )
    norm.fit_transform(X_copy)
    np.testing.assert_array_equal(X_copy, data)


def test_fit_transform_equals_fit_then_transform(data):
    norm_a = RobustConditionalNormalizer(
        categorical_cols=[0, 1], continuous_cols=[2], target_col=3
    )
    norm_b = RobustConditionalNormalizer(
        categorical_cols=[0, 1], continuous_cols=[2], target_col=3
    )
    X_a = norm_a.fit_transform(data)
    X_b = norm_b.fit(data).transform(data)
    np.testing.assert_array_almost_equal(X_a, X_b)


# ---------------------------------------------------------------------------
# Statistical correctness
# ---------------------------------------------------------------------------


def test_zscore_mean_near_zero_per_group(data):
    norm = RobustConditionalNormalizer(
        categorical_cols=[0], continuous_cols=[], target_col=3
    )
    X_norm = norm.fit_transform(data)

    for sex_val in [0, 1]:
        mask = data[:, 0] == sex_val
        group_z = X_norm[mask, 3]
        assert (
            abs(np.mean(group_z)) < 0.3
        ), f"Mean Z-score for sex={sex_val} is {np.mean(group_z):.3f}, expected near 0"


def test_zscore_std_near_one_per_group(data):
    norm = RobustConditionalNormalizer(
        categorical_cols=[0], continuous_cols=[], target_col=3
    )
    X_norm = norm.fit_transform(data)

    for sex_val in [0, 1]:
        mask = data[:, 0] == sex_val
        group_z = X_norm[mask, 3]
        assert (
            0.7 < np.std(group_z) < 1.3
        ), f"Std Z-score for sex={sex_val} is {np.std(group_z):.3f}, expected near 1"


def test_groups_are_normalized_independently(data):
    """After normalization, both groups should have similar scale regardless of
    their original mu/sigma offset."""
    norm = RobustConditionalNormalizer(
        categorical_cols=[0], continuous_cols=[], target_col=3
    )
    X_norm = norm.fit_transform(data)

    means = [np.mean(X_norm[data[:, 0] == v, 3]) for v in [0, 1]]
    stds = [np.std(X_norm[data[:, 0] == v, 3]) for v in [0, 1]]

    # groups had different gamma scales originally — after normalization both near 0
    assert abs(means[0] - means[1]) < 0.5
    assert abs(stds[0] - stds[1]) < 0.4


# ---------------------------------------------------------------------------
# Covariate combinations
# ---------------------------------------------------------------------------


def test_no_categorical_covariates(data):
    norm = RobustConditionalNormalizer(
        categorical_cols=[], continuous_cols=[2], target_col=3
    )
    X_norm = norm.fit_transform(data)
    assert X_norm.shape == data.shape
    assert len(norm._cat_corrections) == 1  # single global group


def test_no_continuous_covariates(data):
    norm = RobustConditionalNormalizer(
        categorical_cols=[0], continuous_cols=[], target_col=3
    )
    X_norm = norm.fit_transform(data)
    assert X_norm.shape == data.shape


def test_no_covariates_global_normalization(data):
    norm = RobustConditionalNormalizer(
        categorical_cols=[], continuous_cols=[], target_col=3
    )
    X_norm = norm.fit_transform(data)
    assert X_norm.shape == data.shape
    assert len(norm._cat_corrections) == 1


def test_two_categorical_covariates(data):
    norm = RobustConditionalNormalizer(
        categorical_cols=[0, 1], continuous_cols=[], target_col=3
    )
    X_norm = norm.fit_transform(data)
    assert X_norm.shape == data.shape
    assert len(norm._cat_corrections[3]) == 2  # (sex=0,batch=0) and (sex=1,batch=1)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_unseen_category_at_transform_warns_and_zeroes(data):
    norm = RobustConditionalNormalizer(
        categorical_cols=[0], continuous_cols=[], target_col=3
    )
    norm.fit(data)

    # Introduce an unseen sex value (sex=99)
    X_unseen = data[:5].copy()
    X_unseen[:, 0] = 99

    with pytest.warns(UserWarning, match="Unseen categorical combination"):
        X_norm = norm.transform(X_unseen)

    np.testing.assert_array_equal(X_norm[:, 3], 0.0)


def test_constant_target_does_not_raise():
    """sigma floor at 1e-6 must prevent division by zero for near-constant data."""
    rng = np.random.default_rng(0)
    X = np.zeros((50, 3))
    X[:, 0] = np.arange(50)  # continuous covariate
    X[:, 1] = 1.0 + rng.gamma(0.001, 0.001, 50)  # near-constant positive target
    norm = RobustConditionalNormalizer(
        categorical_cols=[], continuous_cols=[0], target_col=1
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        X_norm = norm.fit_transform(X)
    assert np.all(np.isfinite(X_norm[:, 1]))


def test_single_sample_per_group_falls_back(rng=RNG):
    """Groups with fewer than MIN_BIN_SAMPLES should fall back gracefully."""
    n = 5
    X = np.column_stack(
        [
            np.zeros(n),  # categorical col (single group)
            rng.uniform(0, 1, n),  # continuous col
            rng.gamma(2, 1, n),  # target: strictly positive for Box-Cox
        ]
    )
    norm = RobustConditionalNormalizer(
        categorical_cols=[0], continuous_cols=[1], target_col=2, n_bins=6
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        X_norm = norm.fit_transform(X)
    assert X_norm.shape == X.shape
    assert np.all(np.isfinite(X_norm[:, 2]))


# ---------------------------------------------------------------------------
# Parameter validation
# ---------------------------------------------------------------------------


def test_too_many_categorical_raises():
    with pytest.raises(ValueError, match="Exceeded max categorical"):
        RobustConditionalNormalizer(
            categorical_cols=[0, 1, 2], continuous_cols=[], target_col=3
        )


def test_too_many_continuous_raises():
    with pytest.raises(ValueError, match="Exceeded max continuous"):
        RobustConditionalNormalizer(
            categorical_cols=[], continuous_cols=[0, 1, 2], target_col=3
        )


def test_two_continuous_covariates():
    rng = np.random.default_rng(7)
    n = 600
    X = np.column_stack([
        rng.integers(0, 2, n).astype(float),
        rng.uniform(20, 80, n),
        rng.uniform(1, 50, n),
        rng.gamma(4.0, 2.0, n),
    ])
    norm = RobustConditionalNormalizer(
        categorical_cols=[0], continuous_cols=[1, 2], target_col=3, n_bins=10
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        X_norm = norm.fit_transform(X)
    assert X_norm.shape == X.shape
    assert np.all(np.isfinite(X_norm[:, 3]))


def test_target_col_in_categorical_raises():
    with pytest.raises(ValueError, match="Target column cannot be included"):
        RobustConditionalNormalizer(
            categorical_cols=[0], continuous_cols=[1], target_col=0
        )


def test_target_col_in_continuous_raises():
    with pytest.raises(ValueError, match="Target column cannot be included"):
        RobustConditionalNormalizer(
            categorical_cols=[0], continuous_cols=[1], target_col=1
        )


# ---------------------------------------------------------------------------
# ContinuousSurfaceFitter unit tests
# ---------------------------------------------------------------------------


def test_surface_fitter_zero_features_uses_fallback(rng=RNG):
    y = rng.gamma(4.0, 2.0, 200)  # strictly positive for Box-Cox
    X_cont = np.empty((200, 0))
    fitter = ContinuousSurfaceFitter(n_bins=6, degree=2)
    fitter.fit(X_cont, y)
    assert not fitter._is_fitted
    mu, sigma = fitter.predict_mu_sigma(X_cont)
    assert mu.shape == (200,)
    assert sigma.shape == (200,)
    assert np.all(sigma > 0)


def test_surface_fitter_predict_before_fit_returns_defaults():
    fitter = ContinuousSurfaceFitter(n_bins=6, degree=2)
    X_cont = np.empty((10, 0))
    mu, sigma = fitter.predict_mu_sigma(X_cont)
    np.testing.assert_array_equal(mu, fitter._global_mu)
    np.testing.assert_array_equal(sigma, fitter._global_sigma)


def test_surface_fitter_sigma_strictly_positive(rng=RNG):
    X_cont = rng.uniform(0, 100, (300, 1))
    y = rng.gamma(2.0, 1.0, 300)  # strictly positive for Box-Cox
    fitter = ContinuousSurfaceFitter(n_bins=6, degree=2)
    fitter.fit(X_cont, y)
    _, sigma = fitter.predict_mu_sigma(X_cont)
    assert np.all(sigma > 0)
