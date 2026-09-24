import numpy as np
import pytest

from ml_principles.metrics import (
    mean_absolute_error,
    mean_baseline,
    mean_squared_error,
    r2_score,
)

METRICS = [
    mean_squared_error,
    mean_absolute_error,
    r2_score,
]


class SparseStub:
    """Enough sparse behavior to test rejection without SciPy."""

    def toarray(self) -> np.ndarray:
        return np.array([1.0, 2.0])

    def tocsr(self) -> "SparseStub":
        return self


def test_known_example() -> None:
    y_true = [1, 2, 3]
    y_pred = [1, 3, 2]

    assert mean_squared_error(y_true, y_pred) == pytest.approx(2 / 3)
    assert mean_absolute_error(y_true, y_pred) == pytest.approx(2 / 3)
    assert r2_score(y_true, y_pred) == pytest.approx(0.0)


def test_perfect_predictions() -> None:
    y_true = [1, 2, 3]
    y_pred = [1, 2, 3]

    assert mean_squared_error(y_true, y_pred) == 0.0
    assert mean_absolute_error(y_true, y_pred) == 0.0
    assert r2_score(y_true, y_pred) == 1.0


def test_poor_prediction_has_expected_negative_r2() -> None:
    y_true = [1, 2, 3]
    y_pred = [10, 10, 10]

    # SS_res = 194
    # SS_tot = 2
    # R² = 1 - 194 / 2 = -96
    assert r2_score(y_true, y_pred) == pytest.approx(-96.0)


@pytest.mark.parametrize("metric", METRICS)
def test_rejects_scalar(metric) -> None:
    with pytest.raises(ValueError):
        metric(1.0, [1.0])


@pytest.mark.parametrize("metric", METRICS)
def test_rejects_two_dimensional(metric) -> None:
    with pytest.raises(ValueError):
        metric([[1, 2], [3, 4]], [[1, 2], [3, 4]])


@pytest.mark.parametrize("metric", METRICS)
def test_rejects_empty(metric) -> None:
    with pytest.raises(ValueError):
        metric([], [])


@pytest.mark.parametrize("metric", METRICS)
def test_rejects_mismatched_lengths(metric) -> None:
    with pytest.raises(ValueError):
        metric([1, 2, 3], [1, 2])


@pytest.mark.parametrize("metric", METRICS)
def test_rejects_boolean_y_true(metric) -> None:
    with pytest.raises(ValueError):
        metric([True, False], [1, 2])


@pytest.mark.parametrize("metric", METRICS)
def test_rejects_boolean_y_pred(metric) -> None:
    with pytest.raises(ValueError):
        metric([1, 2], [True, False])


@pytest.mark.parametrize("metric", METRICS)
def test_rejects_string_y_true(metric) -> None:
    with pytest.raises(ValueError):
        metric(["1", "2"], [1, 2])


@pytest.mark.parametrize("metric", METRICS)
def test_rejects_string_y_pred(metric) -> None:
    with pytest.raises(ValueError):
        metric([1, 2], ["1", "2"])


@pytest.mark.parametrize("metric", METRICS)
def test_rejects_object_array(metric) -> None:
    values = np.array([1, 2], dtype=object)

    with pytest.raises(ValueError):
        metric(values, [1, 2])


@pytest.mark.parametrize("metric", METRICS)
def test_rejects_complex_values(metric) -> None:
    with pytest.raises(ValueError):
        metric([1, 2], [1 + 2j, 3 + 4j])


@pytest.mark.parametrize("metric", METRICS)
@pytest.mark.parametrize(
    "bad_value",
    [np.nan, np.inf, -np.inf],
)
def test_rejects_nonfinite_y_true(metric, bad_value: float) -> None:
    with pytest.raises(ValueError):
        metric([1.0, bad_value], [1.0, 2.0])


@pytest.mark.parametrize("metric", METRICS)
@pytest.mark.parametrize(
    "bad_value",
    [np.nan, np.inf, -np.inf],
)
def test_rejects_nonfinite_y_pred(metric, bad_value: float) -> None:
    with pytest.raises(ValueError):
        metric([1.0, 2.0], [1.0, bad_value])


@pytest.mark.parametrize("metric", METRICS)
def test_rejects_ragged_input(metric) -> None:
    with pytest.raises(ValueError):
        metric([[1], [2, 3]], [1, 2])


@pytest.mark.parametrize("metric", METRICS)
def test_rejects_sparse_input(metric) -> None:
    with pytest.raises(ValueError):
        metric(SparseStub(), [1.0, 2.0])


def test_r2_rejects_single_target() -> None:
    with pytest.raises(ValueError):
        r2_score([1.0], [1.0])


def test_r2_rejects_constant_y_true() -> None:
    with pytest.raises(ValueError):
        r2_score([5.0, 5.0], [4.0, 6.0])


@pytest.mark.parametrize("metric", METRICS)
def test_inputs_are_not_mutated(metric) -> None:
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, 3.0, 2.0])

    expected_true = y_true.copy()
    expected_pred = y_pred.copy()

    metric(y_true, y_pred)

    np.testing.assert_array_equal(y_true, expected_true)
    np.testing.assert_array_equal(y_pred, expected_pred)


@pytest.mark.parametrize(
    "metric",
    [
        mean_squared_error,
        mean_absolute_error,
        r2_score,
    ],
)
def test_rejects_overflow_from_finite_inputs(metric) -> None:
    maximum = np.finfo(np.float64).max

    y_true = np.array([-maximum, maximum])
    y_pred = np.array([maximum, -maximum])

    with pytest.raises(ValueError):
        metric(y_true, y_pred)


def test_baseline_uses_training_mean() -> None:
    y_train = np.array([0.0, 10.0])
    y_eval = np.array([100.0, 200.0, 300.0])

    result = mean_baseline(
        y_train,
        n_predictions=len(y_eval),
    )

    np.testing.assert_array_equal(
        result,
        np.array([5.0, 5.0, 5.0]),
    )


@pytest.mark.parametrize(
    "n_predictions",
    [
        True,
        False,
        2.5,
        "2",
        None,
        0,
        -1,
    ],
)
def test_baseline_rejects_invalid_n_predictions(
    n_predictions: object,
) -> None:
    with pytest.raises(ValueError):
        mean_baseline(
            [1.0, 2.0],
            n_predictions,  # type: ignore[arg-type]
        )


def test_baseline_rejects_nonfinite_training_mean() -> None:
    maximum = np.finfo(np.float64).max

    with pytest.raises(ValueError):
        mean_baseline(
            [maximum, maximum],
            n_predictions=2,
        )


def test_baseline_output_contract() -> None:
    y_train = np.array([1.0, 2.0, 3.0])
    original = y_train.copy()

    result = mean_baseline(
        y_train,
        n_predictions=4,
    )

    assert result.shape == (4,)
    assert result.dtype == np.float64

    np.testing.assert_array_equal(
        result,
        np.array([2.0, 2.0, 2.0, 2.0]),
    )

    np.testing.assert_array_equal(
        y_train,
        original,
    )


def test_metrics_return_python_float() -> None:
    y_true = [1, 2, 3]
    y_pred = [1, 3, 2]

    assert type(mean_squared_error(y_true, y_pred)) is float
    assert type(mean_absolute_error(y_true, y_pred)) is float
    assert type(r2_score(y_true, y_pred)) is float
