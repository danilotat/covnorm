"""
Tests for RobustConditionalNormalizer and ContinuousSurfaceFitter.

Data layout used throughout: [sex, batch, age, marker]
  col 0 - sex        (categorical, values 0/1)
  col 1 - batch      (categorical, values 0/1/2)
  col 2 - age        (continuous)
  col 3 - marker     (target)

Covariates and markers are passed separately to the new API:
  categorical_vals = data[:, [0, 1]]
  continuous_vals  = data[:, [2]]
  X (fit/transform) = data[:, [3]]
"""

import warnings

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.pipeline import Pipeline

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
        categorical_vals=data[:, [0, 1]],
        continuous_vals=data[:, [2]],
    )
    norm.fit(data[:, [3]])
    return norm


# ---------------------------------------------------------------------------
# sklearn estimator contract
# ---------------------------------------------------------------------------


def test_get_params():
    rng = np.random.default_rng(0)
    cat = rng.integers(0, 2, (10, 1)).astype(float)
    cont = rng.uniform(0, 1, (10, 1))
    norm = RobustConditionalNormalizer(
        categorical_vals=cat, continuous_vals=cont, n_bins=8, degree=3
    )
    params = norm.get_params()
    assert "categorical_vals" in params
    assert "continuous_vals" in params
    assert "target_col" not in params
    assert params["n_bins"] == 8
    assert params["degree"] == 3


def test_set_params():
    rng = np.random.default_rng(0)
    cat = rng.integers(0, 2, (10, 1)).astype(float)
    cont = rng.uniform(0, 1, (10, 1))
    norm = RobustConditionalNormalizer(categorical_vals=cat, continuous_vals=cont)
    norm.set_params(n_bins=10, degree=1)
    assert norm.n_bins == 10
    assert norm.degree == 1


def test_clone_produces_unfitted_copy(normalizer_full, data):
    cloned = clone(normalizer_full)
    assert cloned._cat_corrections == {}
    X_norm = cloned.fit_transform(data[:, [3]])
    assert X_norm.shape == data[:, [3]].shape


def test_pipeline_compatible(data):
    pipe = Pipeline(
        [
            (
                "norm",
                RobustConditionalNormalizer(
                    categorical_vals=data[:, [0, 1]],
                    continuous_vals=data[:, [2]],
                ),
            )
        ]
    )
    X_norm = pipe.fit_transform(data[:, [3]])
    assert X_norm.shape == data[:, [3]].shape


# ---------------------------------------------------------------------------
# Output contract
# ---------------------------------------------------------------------------


def test_output_shape(normalizer_full, data):
    X_markers = data[:, [3]]
    X_norm = normalizer_full.transform(X_markers)
    assert X_norm.shape == X_markers.shape


def test_transform_returns_only_target_columns(normalizer_full, data):
    """Key contract: output has marker shape, not full matrix shape."""
    X_markers = data[:, [3]]
    X_norm = normalizer_full.transform(X_markers)
    assert X_norm.shape == X_markers.shape  # (n, 1), not (n, 4)
    assert X_norm.shape[1] == 1


def test_input_array_not_mutated(data):
    X_markers = data[:, [3]].copy()
    X_copy = X_markers.copy()
    norm = RobustConditionalNormalizer(
        categorical_vals=data[:, [0, 1]],
        continuous_vals=data[:, [2]],
    )
    norm.fit_transform(X_markers)
    np.testing.assert_array_equal(X_markers, X_copy)


def test_fit_transform_equals_fit_then_transform(data):
    X_markers = data[:, [3]]
    norm_a = RobustConditionalNormalizer(
        categorical_vals=data[:, [0, 1]],
        continuous_vals=data[:, [2]],
    )
    norm_b = RobustConditionalNormalizer(
        categorical_vals=data[:, [0, 1]],
        continuous_vals=data[:, [2]],
    )
    X_a = norm_a.fit_transform(X_markers)
    X_b = norm_b.fit(X_markers).transform(X_markers)
    np.testing.assert_array_almost_equal(X_a, X_b)


def test_multiple_markers_normalized(data):
    """transform handles multiple marker columns simultaneously."""
    rng = np.random.default_rng(55)
    n = 300
    markers = rng.gamma(2, 5, (n, 3))
    cat = data[:n, [0, 1]]
    cont = data[:n, [2]]
    norm = RobustConditionalNormalizer(categorical_vals=cat, continuous_vals=cont)
    out = norm.fit_transform(markers)
    assert out.shape == markers.shape


def test_transform_with_override_covariates(data):
    """Passing different covariates to transform uses them instead of stored ones."""
    n = data.shape[0]
    norm = RobustConditionalNormalizer(
        categorical_vals=data[:, [0]],
        continuous_vals=data[:, [2]],
    )
    norm.fit(data[:, [3]])
    X_norm = norm.transform(
        data[:, [3]],
        categorical_vals=np.zeros((n, 1)),
        continuous_vals=data[:, [2]],
    )
    assert X_norm.shape == (n, 1)


# ---------------------------------------------------------------------------
# Statistical correctness
# ---------------------------------------------------------------------------


def test_zscore_mean_near_zero_per_group(data):
    norm = RobustConditionalNormalizer(
        categorical_vals=data[:, [0]],
        continuous_vals=np.empty((data.shape[0], 0)),
    )
    X_norm = norm.fit_transform(data[:, [3]])

    for sex_val in [0, 1]:
        mask = data[:, 0] == sex_val
        group_z = X_norm[mask, 0]
        assert (
            abs(np.mean(group_z)) < 0.3
        ), f"Mean Z-score for sex={sex_val} is {np.mean(group_z):.3f}, expected near 0"


def test_zscore_std_near_one_per_group(data):
    norm = RobustConditionalNormalizer(
        categorical_vals=data[:, [0]],
        continuous_vals=np.empty((data.shape[0], 0)),
        anova_alpha=1.0,
    )
    X_norm = norm.fit_transform(data[:, [3]])

    for sex_val in [0, 1]:
        mask = data[:, 0] == sex_val
        group_z = X_norm[mask, 0]
        assert (
            0.7 < np.std(group_z) < 1.3
        ), f"Std Z-score for sex={sex_val} is {np.std(group_z):.3f}, expected near 1"


def test_groups_are_normalized_independently(data):
    """After normalization, both groups should have similar scale regardless of
    their original mu/sigma offset."""
    norm = RobustConditionalNormalizer(
        categorical_vals=data[:, [0]],
        continuous_vals=np.empty((data.shape[0], 0)),
    )
    X_norm = norm.fit_transform(data[:, [3]])

    means = [np.mean(X_norm[data[:, 0] == v, 0]) for v in [0, 1]]
    stds = [np.std(X_norm[data[:, 0] == v, 0]) for v in [0, 1]]

    # groups had different gamma scales originally — after normalization both near 0
    assert abs(means[0] - means[1]) < 0.5
    assert abs(stds[0] - stds[1]) < 0.4


# ---------------------------------------------------------------------------
# Covariate combinations
# ---------------------------------------------------------------------------


def test_no_categorical_covariates(data):
    n = data.shape[0]
    norm = RobustConditionalNormalizer(
        categorical_vals=np.empty((n, 0)),
        continuous_vals=data[:, [2]],
    )
    X_norm = norm.fit_transform(data[:, [3]])
    assert X_norm.shape == data[:, [3]].shape
    assert len(norm._cat_corrections) == 1  # single global group


def test_no_continuous_covariates(data):
    n = data.shape[0]
    norm = RobustConditionalNormalizer(
        categorical_vals=data[:, [0]],
        continuous_vals=np.empty((n, 0)),
    )
    X_norm = norm.fit_transform(data[:, [3]])
    assert X_norm.shape == data[:, [3]].shape


def test_no_covariates_global_normalization(data):
    n = data.shape[0]
    norm = RobustConditionalNormalizer(
        categorical_vals=np.empty((n, 0)),
        continuous_vals=np.empty((n, 0)),
    )
    X_norm = norm.fit_transform(data[:, [3]])
    assert X_norm.shape == data[:, [3]].shape
    assert len(norm._cat_corrections) == 1


def test_two_categorical_covariates(data):
    norm = RobustConditionalNormalizer(
        categorical_vals=data[:, [0, 1]],
        continuous_vals=np.empty((data.shape[0], 0)),
    )
    X_norm = norm.fit_transform(data[:, [3]])
    assert X_norm.shape == data[:, [3]].shape
    assert len(norm._cat_corrections[0]) == 2  # (sex=0,batch=0) and (sex=1,batch=1)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_string_categorical_values(data):
    """String labels ('M'/'F') must be encoded without raising ValueError."""
    rng = np.random.default_rng(99)
    n = data.shape[0]
    cat_str = np.where(rng.integers(0, 2, (n, 1)) == 0, "M", "F")
    cont = data[:, [2]]
    marker = data[:, [3]]
    norm = RobustConditionalNormalizer(categorical_vals=cat_str, continuous_vals=cont)
    X_norm = norm.fit_transform(marker)
    assert X_norm.shape == marker.shape
    assert np.all(np.isfinite(X_norm[:, 0]))


def test_unseen_category_at_transform_warns_and_zeroes(data):
    norm = RobustConditionalNormalizer(
        categorical_vals=data[:, [0]],
        continuous_vals=np.empty((data.shape[0], 0)),
    )
    norm.fit(data[:, [3]])

    unseen_cat = np.full((5, 1), 99.0)
    with pytest.warns(UserWarning, match="Unseen categorical combination"):
        X_norm = norm.transform(
            data[:5, [3]],
            categorical_vals=unseen_cat,
            continuous_vals=np.empty((5, 0)),
        )

    np.testing.assert_array_equal(X_norm[:, 0], 0.0)


def test_constant_target_does_not_raise():
    """sigma floor at 1e-6 must prevent division by zero for near-constant data."""
    rng = np.random.default_rng(0)
    n = 50
    cont = np.arange(n).reshape(-1, 1).astype(float)
    target = 1.0 + rng.gamma(0.001, 0.001, (n, 1))
    norm = RobustConditionalNormalizer(
        categorical_vals=np.empty((n, 0)),
        continuous_vals=cont,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        X_norm = norm.fit_transform(target)
    assert np.all(np.isfinite(X_norm[:, 0]))


def test_single_sample_per_group_falls_back(rng=RNG):
    """Groups with fewer than MIN_BIN_SAMPLES should fall back gracefully."""
    n = 5
    cat = np.zeros((n, 1))
    cont = rng.uniform(0, 1, (n, 1))
    target = rng.gamma(2, 1, (n, 1))
    norm = RobustConditionalNormalizer(
        categorical_vals=cat, continuous_vals=cont, n_bins=6
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        X_norm = norm.fit_transform(target)
    assert X_norm.shape == target.shape
    assert np.all(np.isfinite(X_norm[:, 0]))


# ---------------------------------------------------------------------------
# Parameter validation
# ---------------------------------------------------------------------------


def test_too_many_categorical_raises():
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="Exceeded max categorical"):
        RobustConditionalNormalizer(
            categorical_vals=rng.integers(0, 2, (10, 3)).astype(float),
            continuous_vals=np.empty((10, 0)),
        )


def test_too_many_continuous_raises():
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="Exceeded max continuous"):
        RobustConditionalNormalizer(
            categorical_vals=np.empty((10, 0)),
            continuous_vals=rng.uniform(0, 1, (10, 3)),
        )


def test_two_continuous_covariates():
    rng = np.random.default_rng(7)
    n = 600
    cat = rng.integers(0, 2, (n, 1)).astype(float)
    cont = np.column_stack([rng.uniform(20, 80, n), rng.uniform(1, 50, n)])
    markers = rng.gamma(4.0, 2.0, (n, 1))
    norm = RobustConditionalNormalizer(
        categorical_vals=cat, continuous_vals=cont, n_bins=10
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        X_norm = norm.fit_transform(markers)
    assert X_norm.shape == markers.shape
    assert np.all(np.isfinite(X_norm[:, 0]))


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


# ---------------------------------------------------------------------------
# zero_handles tests
# ---------------------------------------------------------------------------


def _make_data_with_zeros(n: int = 300, rng=None) -> tuple:
    """Return (X_cont, y) where y contains some exact zeros."""
    if rng is None:
        rng = np.random.default_rng(99)
    X_cont = rng.uniform(20, 80, (n, 1))
    y = rng.gamma(2.0, 5.0, n)
    y[rng.choice(n, size=20, replace=False)] = 0.0
    return X_cont, y


def _make_data_with_negatives(n: int = 300, rng=None) -> tuple:
    """Return (X_cont, y) where y contains zeros and negative values."""
    if rng is None:
        rng = np.random.default_rng(77)
    X_cont = rng.uniform(20, 80, (n, 1))
    y = rng.normal(loc=5.0, scale=2.0, size=n)
    return X_cont, y


def test_surface_fitter_eps_with_zeros():
    X_cont, y = _make_data_with_zeros()
    fitter = ContinuousSurfaceFitter(n_bins=6, degree=2, zero_handles="eps")
    fitter.fit(X_cont, y)
    mu, sigma = fitter.predict_mu_sigma(X_cont)
    assert np.all(np.isfinite(mu))
    assert np.all(sigma > 0)


def test_surface_fitter_yeojohnson_with_zeros():
    X_cont, y = _make_data_with_zeros()
    fitter = ContinuousSurfaceFitter(n_bins=6, degree=2, zero_handles="yeojohnson")
    fitter.fit(X_cont, y)
    mu, sigma = fitter.predict_mu_sigma(X_cont)
    assert np.all(np.isfinite(mu))
    assert np.all(sigma > 0)


def test_surface_fitter_yeojohnson_with_negatives():
    X_cont, y = _make_data_with_negatives()
    fitter = ContinuousSurfaceFitter(n_bins=6, degree=2, zero_handles="yeojohnson")
    fitter.fit(X_cont, y)
    mu, sigma = fitter.predict_mu_sigma(X_cont)
    assert np.all(np.isfinite(mu))
    assert np.all(sigma > 0)


def test_normalizer_eps_with_zeros_in_target():
    rng = np.random.default_rng(11)
    n = 300
    cat = rng.integers(0, 2, (n, 1)).astype(float)
    cont = rng.uniform(20, 80, (n, 1))
    y = rng.gamma(2.0, 5.0, (n, 1))
    y[rng.choice(n, size=20, replace=False)] = 0.0
    norm = RobustConditionalNormalizer(
        categorical_vals=cat, continuous_vals=cont, zero_handles="eps"
    )
    X_norm = norm.fit_transform(y)
    assert X_norm.shape == y.shape
    assert np.all(np.isfinite(X_norm[:, 0]))


def test_normalizer_yeojohnson_with_negatives_in_target():
    rng = np.random.default_rng(22)
    n = 300
    cat = rng.integers(0, 2, (n, 1)).astype(float)
    cont = rng.uniform(20, 80, (n, 1))
    y = rng.normal(loc=0.0, scale=3.0, size=(n, 1))
    norm = RobustConditionalNormalizer(
        categorical_vals=cat,
        continuous_vals=cont,
        zero_handles="yeojohnson",
    )
    X_norm = norm.fit_transform(y)
    assert X_norm.shape == y.shape
    assert np.all(np.isfinite(X_norm[:, 0]))


class TestAnovaGating:
    """Significance gates for categorical mean and scale corrections."""

    def test_no_correction_when_groups_identical(self):
        """Same distribution in both groups → f_oneway and levene non-significant → identity."""
        rng = np.random.default_rng(42)
        n = 600
        cat = np.repeat([0.0, 1.0], n // 2).reshape(-1, 1)
        cont = rng.uniform(1, 80, (n, 1))
        marker = rng.gamma(2, 10, (n, 1))

        norm = RobustConditionalNormalizer(
            categorical_vals=cat,
            continuous_vals=cont,
            anova_alpha=0.05,
        )
        norm.fit(marker)

        assert hasattr(norm, "_anova_pvalues_")
        assert hasattr(norm, "_levene_pvalues_")
        assert (
            norm._anova_pvalues_[0] > 0.05
        ), f"Expected non-significant ANOVA, got p={norm._anova_pvalues_[0]}"
        assert (
            norm._levene_pvalues_[0] > 0.05
        ), f"Expected non-significant Levene, got p={norm._levene_pvalues_[0]}"

        for tup, (mu, sigma) in norm._cat_corrections[0].items():
            assert mu == 0.0, f"Expected mu=0.0 for group {tup}, got {mu}"
            assert sigma == 1.0, f"Expected sigma=1.0 for group {tup}, got {sigma}"

    def test_location_and_scale_corrections_applied_when_groups_differ(self):
        """Groups with very different scales trigger both f_oneway and levene → mu and sigma corrected."""
        rng = np.random.default_rng(8)
        n = 600
        cat = np.repeat([0.0, 1.0], n // 2).reshape(-1, 1)
        cont = rng.uniform(1, 80, (n, 1))
        # Large scale ratio creates clear mean and variance differences in z_base
        marker = np.concatenate(
            [
                rng.gamma(2, 5, (n // 2, 1)),
                rng.gamma(2, 30, (n // 2, 1)),
            ]
        )

        norm = RobustConditionalNormalizer(
            categorical_vals=cat,
            continuous_vals=cont,
            anova_alpha=0.05,
        )
        norm.fit(marker)

        assert (
            norm._anova_pvalues_[0] <= 0.05
        ), f"Expected significant f_oneway, got p={norm._anova_pvalues_[0]}"
        assert (
            norm._levene_pvalues_[0] <= 0.05
        ), f"Expected significant Levene, got p={norm._levene_pvalues_[0]}"

        # At least one group must have a non-identity correction on mu or sigma
        corrections = norm._cat_corrections[0]
        any_nonidentity = any(
            mu != 0.0 or sigma != 1.0 for mu, sigma in corrections.values()
        )
        assert any_nonidentity, "Expected at least one non-identity correction"

    def test_anova_alpha_zero_always_skips_both_corrections(self):
        """anova_alpha=0.0 means apply_correction is never True → identity for mu and sigma."""
        rng = np.random.default_rng(9)
        n = 400
        cat = np.repeat([0.0, 1.0], n // 2).reshape(-1, 1)
        cont = rng.uniform(1, 80, (n, 1))
        # Use strongly different groups so that without the guard, corrections would apply
        marker = np.concatenate(
            [
                rng.gamma(2, 5, (n // 2, 1)),
                rng.gamma(2, 30, (n // 2, 1)),
            ]
        )

        norm = RobustConditionalNormalizer(
            categorical_vals=cat,
            continuous_vals=cont,
            anova_alpha=0.0,
        )
        norm.fit(marker)

        # anova_alpha=0.0 uses (alpha > 0.0) and (...) guard → always identity
        for tup, (mu, sigma) in norm._cat_corrections[0].items():
            assert mu == 0.0, f"anova_alpha=0 should skip mu correction, got mu={mu}"
            assert (
                sigma == 1.0
            ), f"anova_alpha=0 should skip sigma correction, got sigma={sigma}"

    def test_degenerate_singleton_group_stored_as_nan(self):
        """A singleton group causes the tests to be skipped; p-values stored as np.nan."""
        rng = np.random.default_rng(11)
        n = 100
        # One group has only 1 sample; f_oneway/levene require >= 2 per group
        cat_vals = np.array([0.0] + [1.0] * (n - 1)).reshape(-1, 1)
        cont = rng.uniform(1, 80, (n, 1))
        marker = rng.gamma(2, 10, (n, 1))

        norm = RobustConditionalNormalizer(
            categorical_vals=cat_vals,
            continuous_vals=cont,
            anova_alpha=0.05,
        )
        norm.fit(marker)

        import math

        assert math.isnan(
            norm._anova_pvalues_[0]
        ), "Expected np.nan for degenerate groups"
        assert math.isnan(
            norm._levene_pvalues_[0]
        ), "Expected np.nan for degenerate groups"
        # All corrections must be identity since tests were skipped
        for tup, (mu, sigma) in norm._cat_corrections[0].items():
            assert mu == 0.0
            assert sigma == 1.0

    def test_anova_alpha_out_of_range_raises(self):
        """anova_alpha outside [0, 1] must raise ValueError at construction."""
        import pytest

        rng = np.random.default_rng(12)
        cat = rng.integers(0, 2, 100).astype(float).reshape(-1, 1)
        cont = rng.uniform(1, 80, (100, 1))
        with pytest.raises(ValueError, match="anova_alpha"):
            RobustConditionalNormalizer(
                categorical_vals=cat,
                continuous_vals=cont,
                anova_alpha=1.5,
            )
        with pytest.raises(ValueError, match="anova_alpha"):
            RobustConditionalNormalizer(
                categorical_vals=cat,
                continuous_vals=cont,
                anova_alpha=-0.1,
            )
