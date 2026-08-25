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
from scipy import stats
from sklearn.base import clone
from sklearn.linear_model import Ridge
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
    assert params["ridge_alpha"] == 0.05
    assert params["anchor_strategy"] == "farthest_point"
    assert params["transform_continuous"] is None


def test_set_params():
    rng = np.random.default_rng(0)
    cat = rng.integers(0, 2, (10, 1)).astype(float)
    cont = rng.uniform(0, 1, (10, 1))
    norm = RobustConditionalNormalizer(categorical_vals=cat, continuous_vals=cont)
    norm.set_params(n_bins=10, degree=1, ridge_alpha=0.1)
    assert norm.n_bins == 10
    assert norm.degree == 1
    assert norm.ridge_alpha == 0.1


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


def test_surface_fitter_fits_log_sigma_targets(monkeypatch):
    rng = np.random.default_rng(420)
    X_cont = rng.uniform(0, 100, (300, 1))
    y = rng.gamma(2.0, 1.0, 300)
    fitter = ContinuousSurfaceFitter(n_bins=6, degree=2, lambda_=0.0, n_iterations=1)
    captured = {}
    original_fit = Ridge.fit

    def capture_sigma_target(model, X, target, *args, **kwargs):
        if model is fitter.sigma_model:
            captured["target"] = np.asarray(target).copy()
        return original_fit(model, X, target, *args, **kwargs)

    monkeypatch.setattr(Ridge, "fit", capture_sigma_target)
    fitter.fit(X_cont, y)

    np.testing.assert_allclose(
        captured["target"], np.log(np.maximum(fitter.bin_sigma_, 1e-6))
    )


def test_surface_fitter_scales_polynomial_features_and_applies_relative_ridge():
    rng = np.random.default_rng(423)
    X_cont = rng.uniform(1, 100, (300, 1))
    y = rng.gamma(2.0, 1.0, 300)
    fitter = ContinuousSurfaceFitter(
        n_bins=6, degree=2, ridge_alpha=0.05, n_iterations=1
    ).fit(X_cont, y)

    X_poly = fitter.poly_transformer.transform(fitter.bin_centers_)
    X_poly_scaled = fitter.poly_scaler.transform(X_poly)
    np.testing.assert_allclose(X_poly_scaled.mean(axis=0), 0.0, atol=1e-12)
    # The bias column is constant and becomes zero; all actual polynomial terms
    # have unit population standard deviation.
    np.testing.assert_allclose(X_poly_scaled[:, 1:].std(axis=0), 1.0)

    effective_alpha = 0.05 * len(fitter.bin_centers_)
    assert fitter.mu_model.alpha == pytest.approx(effective_alpha)
    assert fitter.sigma_model.alpha == pytest.approx(effective_alpha)


def test_surface_fitter_floors_log_sigma_before_exponentiating():
    rng = np.random.default_rng(421)
    X_cont = rng.uniform(1, 100, (300, 1))
    y = rng.gamma(2.0, 1.0, 300)
    fitter = ContinuousSurfaceFitter(n_bins=6, degree=2).fit(X_cont, y)

    # A very negative constant log-scale sits below every final fitting window.
    fitter.sigma_model.coef_ = np.zeros_like(fitter.sigma_model.coef_)
    fitter.sigma_model.intercept_ = -1e6
    _, sigma = fitter.predict_mu_sigma(X_cont)

    np.testing.assert_allclose(sigma, np.min(fitter.bin_sigma_))


def test_surface_fitter_caps_log_sigma_before_exponentiating():
    rng = np.random.default_rng(422)
    X_cont = rng.uniform(1, 100, (300, 1))
    y = rng.gamma(2.0, 1.0, 300)
    fitter = ContinuousSurfaceFitter(n_bins=6, degree=2).fit(X_cont, y)

    fitter.sigma_model.coef_ = np.zeros_like(fitter.sigma_model.coef_)
    fitter.sigma_model.intercept_ = 1e6
    _, sigma = fitter.predict_mu_sigma(X_cont)

    np.testing.assert_allclose(sigma, np.max(fitter.bin_sigma_))


# ---------------------------------------------------------------------------
# Continuous-covariate transformations
# ---------------------------------------------------------------------------


def _make_two_covariate_surface_data(n=600, seed=123):
    rng = np.random.default_rng(seed)
    week = rng.uniform(24.0, 42.0, n)
    weight = 180.0 * week - 3500.0 + rng.normal(0.0, 250.0, n)
    X_cont = np.column_stack([weight, week])
    weight_z = (weight - weight.mean()) / weight.std()
    week_z = (week - week.mean()) / week.std()
    y = np.exp(2.0 + 0.25 * weight_z - 0.15 * week_z + rng.normal(0.0, 0.2, n))
    return X_cont, y


def test_ridge_stabilizes_correlated_degree_three_surface():
    X_cont, y = _make_two_covariate_surface_data(seed=321)
    kwargs = dict(
        n_bins=30,
        degree=3,
        lambda_=0.0,
        n_iterations=1,
        transform_continuous="zscore",
    )
    unregularized = ContinuousSurfaceFitter(ridge_alpha=0.0, **kwargs).fit(X_cont, y)
    regularized = ContinuousSurfaceFitter(ridge_alpha=0.05, **kwargs).fit(X_cont, y)

    def unsupported_log_scales(fitter):
        X_transformed = fitter._transform_continuous_covariates(X_cont, fit=False)
        X_poly = fitter.poly_transformer.transform(X_transformed)
        X_poly_scaled = fitter.poly_scaler.transform(X_poly)
        log_sigma = fitter.sigma_model.predict(X_poly_scaled)
        lower, upper = fitter._log_sigma_bounds_
        return int(np.count_nonzero((log_sigma < lower) | (log_sigma > upper)))

    assert unsupported_log_scales(regularized) < unsupported_log_scales(unregularized)


def test_zscore_transform_uses_training_statistics_and_reuses_them_for_prediction():
    X_cont, y = _make_two_covariate_surface_data()
    fitter = ContinuousSurfaceFitter(
        n_bins=30,
        degree=2,
        lambda_=0.0,
        n_iterations=1,
        transform_continuous="zscore",
    ).fit(X_cont, y)

    np.testing.assert_allclose(fitter.continuous_center_, X_cont.mean(axis=0))
    np.testing.assert_allclose(fitter.continuous_scale_, X_cont.std(axis=0))

    X_scaled = fitter._transform_continuous_covariates(X_cont, fit=False)
    np.testing.assert_allclose(X_scaled.mean(axis=0), 0.0, atol=1e-14)
    np.testing.assert_allclose(X_scaled.std(axis=0), 1.0, atol=1e-14)

    # A row must receive the same transformation and prediction by itself as
    # it does inside a larger batch: prediction must not refit the scaler.
    mu_one, sigma_one = fitter.predict_mu_sigma(X_cont[[0]])
    mu_batch, sigma_batch = fitter.predict_mu_sigma(X_cont[:20])
    np.testing.assert_allclose(mu_one, mu_batch[[0]])
    np.testing.assert_allclose(sigma_one, sigma_batch[[0]])


def test_normalizer_forwards_zscore_transform_to_marker_fitters():
    X_cont, y = _make_two_covariate_surface_data(seed=321)
    normalizer = RobustConditionalNormalizer(
        categorical_vals=np.empty((len(y), 0)),
        continuous_vals=X_cont,
        n_bins=30,
        degree=2,
        n_iterations=1,
        transform_continuous="zscore",
    )

    z = normalizer.fit_transform(y.reshape(-1, 1))
    fitter = normalizer._fitters[0]
    assert fitter._resolved_transform_continuous_ == "zscore"
    np.testing.assert_allclose(fitter.continuous_center_, X_cont.mean(axis=0))
    np.testing.assert_allclose(fitter.continuous_scale_, X_cont.std(axis=0))
    assert np.all(np.isfinite(z))


def test_zscore_transform_is_invariant_to_positive_affine_units():
    X_cont, y = _make_two_covariate_surface_data(seed=456)
    X_other_units = X_cont * np.array([0.001, 10.0]) + np.array([2.0, -100.0])

    kwargs = dict(
        n_bins=30,
        degree=2,
        lambda_=0.0,
        n_iterations=1,
        transform_continuous="zscore",
    )
    original = ContinuousSurfaceFitter(**kwargs).fit(X_cont, y)
    converted = ContinuousSurfaceFitter(**kwargs).fit(X_other_units, y)

    mu_original, sigma_original = original.predict_mu_sigma(X_cont)
    mu_converted, sigma_converted = converted.predict_mu_sigma(X_other_units)
    np.testing.assert_allclose(mu_original, mu_converted, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(sigma_original, sigma_converted, rtol=1e-10, atol=1e-10)


def test_transform_continuous_log10_matches_legacy_boolean():
    rng = np.random.default_rng(789)
    X_cont = rng.uniform(1.0, 1000.0, (400, 1))
    y = rng.gamma(3.0, 2.0, 400)
    kwargs = dict(n_bins=8, degree=2, lambda_=0.2, n_iterations=1)

    with pytest.warns(FutureWarning, match="removed in covnorm 1.x") as caught:
        legacy = ContinuousSurfaceFitter(**kwargs, log_transform_continuous=True).fit(
            X_cont, y
        )
    assert len(caught) == 1

    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        current = ContinuousSurfaceFitter(**kwargs, transform_continuous="log10").fit(
            X_cont, y
        )

    np.testing.assert_allclose(legacy.bin_centers_, current.bin_centers_)
    mu_legacy, sigma_legacy = legacy.predict_mu_sigma(X_cont)
    mu_current, sigma_current = current.predict_mu_sigma(X_cont)
    np.testing.assert_allclose(mu_legacy, mu_current)
    np.testing.assert_allclose(sigma_legacy, sigma_current)


def test_normalizer_warns_once_for_legacy_log_alias_across_multiple_markers():
    rng = np.random.default_rng(790)
    n = 400
    cont = rng.uniform(1.0, 1000.0, (n, 1))
    markers = rng.gamma(3.0, 2.0, (n, 2))
    normalizer = RobustConditionalNormalizer(
        categorical_vals=np.empty((n, 0)),
        continuous_vals=cont,
        n_bins=8,
        degree=2,
        n_iterations=1,
        log_transform_continuous=True,
    )

    with pytest.warns(FutureWarning, match="removed in covnorm 1.x") as caught:
        normalizer.fit(markers)

    assert len(caught) == 1
    assert all(
        fitter._resolved_transform_continuous_ == "log10"
        for fitter in normalizer._fitters.values()
    )
    assert all(
        not fitter.log_transform_continuous for fitter in normalizer._fitters.values()
    )


def test_transform_continuous_validation_and_legacy_conflict():
    cat = np.empty((10, 0))
    cont = np.arange(10.0).reshape(-1, 1)

    with pytest.raises(ValueError, match="transform_continuous must be one of"):
        RobustConditionalNormalizer(
            categorical_vals=cat,
            continuous_vals=cont,
            transform_continuous="standardize",
        )

    with pytest.raises(ValueError, match="conflicts"):
        RobustConditionalNormalizer(
            categorical_vals=cat,
            continuous_vals=cont,
            log_transform_continuous=True,
            transform_continuous="zscore",
        )

    with pytest.raises(ValueError, match="ridge_alpha"):
        RobustConditionalNormalizer(
            categorical_vals=cat,
            continuous_vals=cont,
            ridge_alpha=-0.1,
        )


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


def test_percentile_fit_excludes_zeros_and_logs_training_prevalence(caplog):
    rng = np.random.default_rng(20260825)
    n = 400
    X_cont = rng.uniform(20.0, 80.0, (n, 1))
    y = rng.gamma(2.0, 5.0, n)
    zero_mask = np.zeros(n, dtype=bool)
    zero_mask[rng.choice(n, size=80, replace=False)] = True
    y[zero_mask] = 0.0

    normalizer = RobustConditionalNormalizer(
        categorical_vals=np.empty((n, 0)),
        continuous_vals=X_cont,
        n_bins=6,
        degree=2,
        n_iterations=1,
        bin_size=40,
    )
    with caplog.at_level("INFO", logger="covnorm._normalizer"):
        normalizer.fit(y.reshape(-1, 1))

    expected_p0 = 80 / n
    expected_zero_z = stats.norm.ppf(expected_p0 / 2.0)
    assert normalizer.zero_counts_[0] == 80
    assert normalizer.zero_fractions_[0] == pytest.approx(expected_p0)
    assert normalizer.zero_zscores_[0] == pytest.approx(expected_zero_z)
    assert "excluding 80/400 zero values" in caplog.text
    assert "Phi^-1(p0/2)" in caplog.text

    fitter = normalizer._fitters[0]
    assert fitter.n_samples_seen_ == n
    assert fitter.zero_count_ == 80
    assert fitter.positive_count_ == 320

    positive_fitter = ContinuousSurfaceFitter(
        n_bins=6,
        degree=2,
        n_iterations=1,
        bin_size=40,
    ).fit(X_cont[~zero_mask], y[~zero_mask])
    assert fitter.lambda_ == positive_fitter.lambda_
    mu, sigma = fitter.predict_mu_sigma(X_cont)
    positive_mu, positive_sigma = positive_fitter.predict_mu_sigma(X_cont)
    np.testing.assert_allclose(mu, positive_mu)
    np.testing.assert_allclose(sigma, positive_sigma)


def test_percentile_transform_maps_zero_mass_and_positive_distribution():
    rng = np.random.default_rng(101)
    n = 300
    X_cont = rng.uniform(20.0, 80.0, (n, 1))
    y = rng.gamma(2.0, 5.0, n)
    zero_mask = np.zeros(n, dtype=bool)
    zero_mask[:60] = True
    y[zero_mask] = 0.0

    normalizer = RobustConditionalNormalizer(
        categorical_vals=np.empty((n, 0)),
        continuous_vals=X_cont,
        n_bins=6,
        degree=2,
        n_iterations=1,
        bin_size=40,
    ).fit(y.reshape(-1, 1))
    transformed = normalizer.transform(y.reshape(-1, 1))[:, 0]

    p0 = 60 / n
    expected_zero_z = stats.norm.ppf(p0 / 2.0)
    np.testing.assert_allclose(transformed[zero_mask], expected_zero_z)

    fitter = normalizer._fitters[0]
    y_bc = fitter._transform(y[~zero_mask], fitter.lambda_)
    mu, sigma = fitter.predict_mu_sigma(X_cont[~zero_mask])
    positive_z = (y_bc - mu) / np.maximum(sigma, 1e-6)
    expected_positive = stats.norm.ppf(p0 + (1.0 - p0) * stats.norm.cdf(positive_z))
    np.testing.assert_allclose(transformed[~zero_mask], expected_positive)
    assert np.all(transformed[~zero_mask] > expected_zero_z)
    assert np.all(np.isfinite(transformed))


def test_percentile_transform_reuses_training_p0_for_new_batch():
    rng = np.random.default_rng(102)
    n = 300
    train_cont = rng.uniform(20.0, 80.0, (n, 1))
    train_y = rng.gamma(2.0, 5.0, n)
    train_y[:30] = 0.0
    normalizer = RobustConditionalNormalizer(
        categorical_vals=np.empty((n, 0)),
        continuous_vals=train_cont,
        n_bins=6,
        degree=2,
        n_iterations=1,
        bin_size=40,
    ).fit(train_y.reshape(-1, 1))

    new_n = 40
    new_cont = rng.uniform(20.0, 80.0, (new_n, 1))
    new_y = rng.gamma(2.0, 5.0, new_n)
    new_y[:20] = 0.0
    transformed = normalizer.transform(
        new_y.reshape(-1, 1),
        categorical_vals=np.empty((new_n, 0)),
        continuous_vals=new_cont,
    )[:, 0]

    expected_from_training = stats.norm.ppf((30 / n) / 2.0)
    np.testing.assert_allclose(transformed[:20], expected_from_training)


def test_percentile_transform_rejects_zero_unseen_during_fit():
    rng = np.random.default_rng(103)
    n = 200
    train_cont = rng.uniform(20.0, 80.0, (n, 1))
    train_y = rng.gamma(2.0, 5.0, (n, 1))
    normalizer = RobustConditionalNormalizer(
        categorical_vals=np.empty((n, 0)),
        continuous_vals=train_cont,
        n_bins=6,
        degree=2,
        n_iterations=1,
        bin_size=40,
    ).fit(train_y)

    new_y = np.array([[0.0], [1.0]])
    new_cont = np.array([[30.0], [40.0]])
    with pytest.raises(ValueError, match="no zeros were observed during fit"):
        normalizer.transform(
            new_y,
            categorical_vals=np.empty((2, 0)),
            continuous_vals=new_cont,
        )


def test_percentile_fit_requires_two_positive_values():
    X_cont = np.arange(20.0).reshape(-1, 1)
    y = np.zeros(20)
    y[-1] = 1.0
    fitter = ContinuousSurfaceFitter(n_bins=6)

    with pytest.raises(ValueError, match="at least two positive"):
        fitter.fit(X_cont, y)


def test_percentile_categorical_corrections_ignore_zeros():
    rng = np.random.default_rng(104)
    n = 400
    cat = np.repeat([0.0, 1.0], n // 2).reshape(-1, 1)
    cont = rng.uniform(20.0, 80.0, (n, 1))
    y = rng.gamma(2.0, 5.0, (n, 1))
    zero_mask = np.zeros(n, dtype=bool)
    zero_mask[:80] = True
    y[zero_mask] = 0.0

    with_zeros = RobustConditionalNormalizer(
        categorical_vals=cat,
        continuous_vals=cont,
        n_bins=6,
        degree=2,
        n_iterations=1,
        bin_size=40,
        anova_alpha=1.0,
    ).fit(y)
    positives_only = RobustConditionalNormalizer(
        categorical_vals=cat[~zero_mask],
        continuous_vals=cont[~zero_mask],
        n_bins=6,
        degree=2,
        n_iterations=1,
        bin_size=40,
        anova_alpha=1.0,
    ).fit(y[~zero_mask])

    for group, expected in positives_only._cat_corrections[0].items():
        np.testing.assert_allclose(with_zeros._cat_corrections[0][group], expected)


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
        # Construct differences that remain in both location and scale after the
        # marker transformation. A pure Gamma scale-family difference can be
        # largely removed by Box-Cox and is not a reliable Levene fixture.
        marker = np.concatenate(
            [
                np.exp(rng.normal(0.0, 0.2, (n // 2, 1))),
                np.exp(rng.normal(1.0, 1.0, (n // 2, 1))),
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


def test_fit_survives_outlier_mask_collapsing_to_empty(monkeypatch):
    """Regression test for a crash in the iterative outlier-rejection loop.

    A near-degenerate target (e.g. a zero-inflated marker, >99% identical
    values) can drive every bin's sigma estimate down to the 1e-6 floor
    after the first pruning pass. Any leftover residual then produces
    Z-scores in the tens of thousands, and the |Z| <= 3.372 mask can reject
    every remaining sample. The next iteration used to call
    `_create_knn_bins` on a zero-row array and crash with IndexError.
    `fit` must degrade gracefully instead (see MASKCOLLAPSE.md).
    """
    rng = np.random.default_rng(0)
    n = 1000
    # 99% of samples share the same near-zero floor value; the rest are
    # scattered small positive values (mimics a below-detection-limit marker).
    y = np.where(rng.random(n) < 0.99, 1e-3, rng.integers(1, 20, n) / 100.0)
    X = np.column_stack(
        [
            rng.integers(2600, 4500, n).astype(float),  # weight-like covariate
            rng.integers(38, 43, n).astype(float),  # gestational-week-like covariate
        ]
    )
    fitter = ContinuousSurfaceFitter(
        n_bins=20, degree=3, n_iterations=3, bin_size=200, zero_handles="eps"
    )

    # Keep this regression independent of the surface estimator: Ridge can
    # legitimately avoid the natural collapse that originally exposed the bug.
    # A uniformly tiny predicted scale deterministically exercises the same
    # all-false outlier mask and proves the loop still stops before an empty refit.
    monkeypatch.setattr(
        fitter,
        "_predict_sigma_from_poly",
        lambda X_poly: np.full(X_poly.shape[0], 1e-6),
    )

    with pytest.warns(
        UserWarning, match="Outlier rejection would remove all remaining samples"
    ):
        fitter.fit(X, y)  # must not raise IndexError

    assert fitter._is_fitted
    assert fitter.bin_centers_ is not None


@pytest.mark.parametrize("strategy", ["farthest_point", "projection_rank"])
def test_create_knn_bins_empty_input_returns_empty_lists(strategy):
    """_create_knn_bins must degrade gracefully on zero rows, matching
    _create_rolling_bins' existing behaviour for the same input, instead of
    raising IndexError from indexing an empty array. The zero-row guard runs
    ahead of anchor selection, so it holds for either strategy.
    """
    fitter = ContinuousSurfaceFitter(n_bins=20, bin_size=200, anchor_strategy=strategy)
    fitter.lambda_ = 0.4

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        centers, mus, sigmas = fitter._create_knn_bins(np.empty((0, 2)), np.empty(0))

    assert centers == []
    assert mus == []
    assert sigmas == []


# ---------------------------------------------------------------------------
# k-NN window anchor selection (anchor_strategy)
# ---------------------------------------------------------------------------


def _ridge_and_off_curve_cloud():
    """Already-scaled 2D cloud: a dense correlated ridge plus a sparse
    off-ridge population. Mimics a growth curve (e.g. gestational week x birth
    weight) with a genuine but under-represented off-curve group.

    Returns (X_scaled, n_ridge); rows with index >= n_ridge are off-ridge.
    """
    rng = np.random.default_rng(11)
    n_ridge, n_off = 800, 40
    t = rng.uniform(-1.7, 1.7, n_ridge)
    ridge = np.column_stack([t, t + rng.normal(0, 0.05, n_ridge)])
    off = np.column_stack(
        [rng.uniform(0.8, 1.8, n_off), rng.uniform(-1.8, -0.8, n_off)]
    )
    return np.vstack([ridge, off]), n_ridge


def _min_pairwise_distance(points: np.ndarray) -> float:
    d = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=-1)
    return float(d[~np.eye(len(points), dtype=bool)].min())


def test_farthest_point_anchors_reach_off_curve_observations():
    """The reason farthest-point sampling exists: evenly spaced ranks are
    evenly spaced in probability mass, so on a correlated covariate pair every
    projection_rank anchor lands on the dense ridge and the polynomial surface
    is unidentified off it. FPS spreads the anchors over the occupied region.
    """
    X_scaled, n_ridge = _ridge_and_off_curve_cloud()

    rank_idx = ContinuousSurfaceFitter(
        n_bins=30, anchor_strategy="projection_rank"
    )._select_reference_points(X_scaled)
    fps_idx = ContinuousSurfaceFitter(
        n_bins=30, anchor_strategy="farthest_point"
    )._select_reference_points(X_scaled)

    assert int((rank_idx >= n_ridge).sum()) == 0
    assert int((fps_idx >= n_ridge).sum()) > 0
    assert _min_pairwise_distance(X_scaled[fps_idx]) > _min_pairwise_distance(
        X_scaled[rank_idx]
    )


def test_projection_rank_reproduces_previous_anchor_selection():
    """projection_rank must stay bit-identical to the pre-FPS selection so the
    old behaviour really is reachable."""
    X_scaled, _ = _ridge_and_off_curve_cloud()
    n, n_bins = X_scaled.shape[0], 30

    expected = np.unique(
        np.argsort(X_scaled.sum(axis=1))[
            np.round(np.linspace(0, n - 1, n_bins)).astype(int)
        ]
    )
    got = ContinuousSurfaceFitter(
        n_bins=n_bins, anchor_strategy="projection_rank"
    )._select_reference_points(X_scaled)

    np.testing.assert_array_equal(got, expected)


def test_farthest_point_selection_is_deterministic():
    """Greedy FPS uses no RNG and argmin/argmax break ties by first index, so
    repeated selections must be identical."""
    X_scaled, _ = _ridge_and_off_curve_cloud()
    first = ContinuousSurfaceFitter(
        n_bins=30, anchor_strategy="farthest_point"
    )._select_reference_points(X_scaled)
    second = ContinuousSurfaceFitter(
        n_bins=30, anchor_strategy="farthest_point"
    )._select_reference_points(X_scaled)
    np.testing.assert_array_equal(first, second)


def test_farthest_point_anchors_are_observed_rows_and_capped():
    """Anchors are real rows -- never synthetic points in an empty region --
    and are capped at n_bins, degrading to at most n when rows are scarce."""
    X_scaled, _ = _ridge_and_off_curve_cloud()
    n = X_scaled.shape[0]

    idx = ContinuousSurfaceFitter(
        n_bins=30, anchor_strategy="farthest_point"
    )._select_reference_points(X_scaled)
    assert len(idx) == 30
    assert idx.min() >= 0 and idx.max() < n

    scarce = ContinuousSurfaceFitter(
        n_bins=30, anchor_strategy="farthest_point"
    )._select_reference_points(X_scaled[:5])
    assert len(scarce) <= 5
    assert len(np.unique(scarce)) == len(scarce)


def test_unknown_anchor_strategy_raises():
    rng = np.random.default_rng(3)
    X = np.column_stack([rng.uniform(20, 80, 300), rng.uniform(1, 50, 300)])
    y = rng.gamma(4.0, 2.0, 300)
    fitter = ContinuousSurfaceFitter(n_bins=10, anchor_strategy="nearest_point")
    with pytest.raises(ValueError, match="anchor_strategy"):
        fitter.fit(X, y)


def test_two_continuous_covariates_farthest_point_surface():
    """End-to-end on the motivating shape (correlated covariate pair) with the
    documented degree=2 / n_bins=30 pairing, and proof that the normalizer
    forwards anchor_strategy to its per-marker fitters."""
    rng = np.random.default_rng(5)
    n = 900
    week = rng.uniform(37.0, 42.0, n)
    weight = 200.0 * week + rng.normal(0, 150.0, n)  # follows week: a growth curve
    cont = np.column_stack([week, weight])
    cat = rng.integers(0, 2, (n, 1)).astype(float)
    markers = rng.gamma(4.0, 2.0, (n, 1))

    norm = RobustConditionalNormalizer(
        categorical_vals=cat,
        continuous_vals=cont,
        n_bins=30,
        degree=2,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        X_norm = norm.fit_transform(markers)

    assert X_norm.shape == markers.shape
    assert np.all(np.isfinite(X_norm[:, 0]))
    fitter = norm._fitters[0]
    assert fitter.anchor_strategy == "farthest_point"
    assert fitter._is_fitted
    assert fitter.bin_centers_.shape == (30, 2)


# ---------------------------------------------------------------------------
# Regression: a single pathological reference row must not blind a marker
# ---------------------------------------------------------------------------


def test_marginal_estimates_are_available_after_a_successful_surface_fit(rng=RNG):
    """`marginal_mu_`/`marginal_sigma_` are the degradation target when the
    polynomial surface turns out to be invalid at some covariate value, so they
    must be populated even when the surface itself fitted fine (the
    `_global_*` pair is only meaningful on the `_fit_fallback` path)."""
    X = rng.uniform(0, 100, (300, 1))
    y = rng.gamma(2.0, 1.0, 300)
    fitter = ContinuousSurfaceFitter(n_bins=6, degree=2)
    fitter.fit(X, y)

    assert fitter._is_fitted
    assert np.isfinite(fitter.marginal_mu_)
    assert fitter.marginal_sigma_ > 0


def test_predict_mu_sigma_substitutes_marginal_when_log_sigma_is_non_finite(rng=RNG):
    """A non-finite log-scale is unsupported and degrades to the marginal fit."""
    X = rng.uniform(0, 100, (300, 1))
    y = rng.gamma(2.0, 1.0, 300)
    fitter = ContinuousSurfaceFitter(n_bins=6, degree=2)
    fitter.fit(X, y)

    fitter.sigma_model.intercept_ = np.nan

    with pytest.warns(UserWarning, match="invalid mu or sigma"):
        mu, sigma = fitter.predict_mu_sigma(X)

    assert np.all(sigma > 0)
    np.testing.assert_allclose(sigma, fitter.marginal_sigma_)
    # mu from the same invalid surface is not trustworthy either
    np.testing.assert_allclose(mu, fitter.marginal_mu_)


def test_categorical_correction_survives_one_extreme_reference_row(monkeypatch):
    """`sigma_cat` is estimated from the reference `z_base`, so a non-robust
    np.std lets a single extreme row inflate it by orders of magnitude. Every
    sample for that column is then divided by the inflated value and the column
    silently goes blind -- it can no longer flag anything.

    In practice the extreme `z_base` comes from an unsupported covariate corner
    where the scale surface is invalid, so it is injected here directly: this
    pins the correction as robust independently of whatever produced the outlier.
    anova_alpha=1.0 forces both corrections on, exercising the estimator rather
    than the gating."""
    rng = np.random.default_rng(3)
    n = 600
    cat = np.repeat([0.0, 1.0], n // 2).reshape(-1, 1)
    cont = rng.uniform(1, 80, (n, 1))
    marker = rng.gamma(2, 10, (n, 1))

    unpatched = ContinuousSurfaceFitter.predict_mu_sigma

    def one_invalid_scale(self, X_cont):
        mu, sigma = unpatched(self, X_cont)
        if len(sigma) == n:  # the reference pass, not a later transform
            sigma = sigma.copy()
            sigma[0] = 1e-6
        return mu, sigma

    monkeypatch.setattr(ContinuousSurfaceFitter, "predict_mu_sigma", one_invalid_scale)

    norm = RobustConditionalNormalizer(
        categorical_vals=cat, continuous_vals=cont, anova_alpha=1.0
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        norm.fit(marker)

    _, sigma_cat = norm._cat_corrections[0][(0.0,)]
    assert sigma_cat < 5.0, f"one row inflated sigma_cat to {sigma_cat:.4g}"

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        z = norm.transform(marker, categorical_vals=cat, continuous_vals=cont)

    clean = z[1 : n // 2, 0]  # the untouched rows of the poisoned group
    assert clean.std() > 0.3, f"marker went blind: z-score std={clean.std():.4g}"
