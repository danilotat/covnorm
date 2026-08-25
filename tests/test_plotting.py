import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.collections import PathCollection
from scipy import stats

from covnorm import RobustConditionalNormalizer, plot_worm
from covnorm._plotting import _worm_statistics


def _fit_one_covariate_normalizer(n: int = 240):
    rng = np.random.default_rng(2026)
    age = rng.uniform(20.0, 80.0, (n, 1))
    latent = rng.normal(size=n)
    marker = np.exp(1.0 + 0.015 * age[:, 0] + 0.3 * latent).reshape(-1, 1)
    normalizer = RobustConditionalNormalizer(
        categorical_vals=np.empty((n, 0)),
        continuous_vals=age,
        n_bins=6,
        degree=2,
        n_iterations=1,
        bin_size=40,
    ).fit(marker)
    return normalizer, marker, age


def test_plot_worm_draws_equal_count_conditional_panels():
    normalizer, marker, age = _fit_one_covariate_normalizer()

    fig = plot_worm(
        normalizer,
        marker,
        n_bins=4,
        covariate_label="age",
        marker_label="CD4",
    )

    assert len(fig.axes) == 4
    assert all("age:" in ax.get_title() for ax in fig.axes)
    assert fig._suptitle.get_text() == "Worm plot — CD4"

    z_scores = normalizer.transform(marker)[:, 0]
    first_group = np.array_split(np.argsort(age[:, 0], kind="mergesort"), 4)[0]
    observed = np.sort(z_scores[first_group])
    n = len(observed)
    probabilities = (np.arange(1, n + 1) - 3.0 / 8.0) / (n + 1.0 / 4.0)
    theoretical = stats.norm.ppf(probabilities)
    expected = np.column_stack([theoretical, observed - theoretical])

    points = next(
        collection
        for collection in fig.axes[0].collections
        if isinstance(collection, PathCollection)
    )
    np.testing.assert_allclose(points.get_offsets(), expected)

    summary = fig.axes[0].texts[0].get_text()
    assert "median=" in summary
    assert "MADσ=" in summary
    assert "tail−=" in summary
    assert "tail+=" in summary
    plt.close(fig)


def test_plot_worm_accepts_covariate_overrides_for_new_samples():
    normalizer, _, _ = _fit_one_covariate_normalizer()
    rng = np.random.default_rng(7)
    new_age = rng.uniform(20.0, 80.0, (60, 1))
    new_marker = np.exp(
        1.0 + 0.015 * new_age[:, 0] + 0.3 * rng.normal(size=60)
    ).reshape(-1, 1)

    fig = plot_worm(
        normalizer,
        new_marker,
        categorical_vals=np.empty((60, 0)),
        continuous_vals=new_age,
        n_bins=3,
    )

    assert len(fig.axes) == 3
    plt.close(fig)


def test_plot_worm_without_continuous_covariates_draws_one_panel():
    rng = np.random.default_rng(11)
    marker = rng.lognormal(size=(120, 1))
    normalizer = RobustConditionalNormalizer(
        categorical_vals=np.empty((120, 0)),
        continuous_vals=np.empty((120, 0)),
        n_iterations=1,
    ).fit(marker)

    fig = plot_worm(normalizer, marker, n_bins=5)

    assert len(fig.axes) == 1
    assert fig.axes[0].get_title() == "All observations (n=120)"
    plt.close(fig)


def test_plot_worm_can_condition_on_second_continuous_covariate():
    rng = np.random.default_rng(13)
    n = 240
    age = rng.uniform(20.0, 80.0, n)
    bmi = rng.uniform(18.0, 35.0, n)
    continuous = np.column_stack([age, bmi])
    marker = np.exp(0.8 + 0.01 * age + 0.025 * bmi + 0.25 * rng.normal(size=n)).reshape(
        -1, 1
    )
    normalizer = RobustConditionalNormalizer(
        categorical_vals=np.empty((n, 0)),
        continuous_vals=continuous,
        n_bins=10,
        degree=2,
        n_iterations=1,
        bin_size=40,
        transform_continuous="zscore",
    ).fit(marker)

    fig = plot_worm(
        normalizer,
        marker,
        covariate_index=1,
        covariate_label="BMI",
        n_bins=4,
    )

    assert len(fig.axes) == 4
    assert all("BMI:" in ax.get_title() for ax in fig.axes)
    plt.close(fig)


def test_worm_statistics_report_robust_scale_and_both_tails():
    residuals = np.array([-4.0, -1.0, 0.0, 1.0, 5.0])

    median, mad_scale, lower_tail, upper_tail = _worm_statistics(residuals)

    assert median == 0.0
    assert mad_scale == pytest.approx(1.0 / stats.norm.ppf(0.75))
    assert lower_tail == pytest.approx(0.2)
    assert upper_tail == pytest.approx(0.2)


@pytest.mark.parametrize(
    ("kwargs", "exception", "message"),
    [
        ({"n_bins": 0}, ValueError, "n_bins"),
        ({"alpha": 1.0}, ValueError, "alpha"),
        ({"marker_col": 2}, IndexError, "marker_col"),
        ({"covariate_index": 1}, IndexError, "covariate_index"),
    ],
)
def test_plot_worm_validates_plot_parameters(kwargs, exception, message):
    normalizer, marker, _ = _fit_one_covariate_normalizer()

    with pytest.raises(exception, match=message):
        plot_worm(normalizer, marker, **kwargs)
