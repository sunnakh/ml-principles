"""Behavioral contract for dense feature standardization."""

import numpy as np
import pytest

from ml_principles.preprocessing import StandardScaler


def test_fit_learns_population_statistics_and_returns_self() -> None:
    scaler = StandardScaler()
    assert scaler.fit([[1, 10], [3, 14]]) is scaler
    np.testing.assert_array_equal(scaler.mean_, [2, 12])
    np.testing.assert_array_equal(scaler.var_, [1, 4])
    np.testing.assert_array_equal(scaler.scale_, [1, 2])
    assert scaler.n_features_in_ == 2
    for attribute in (scaler.mean_, scaler.var_, scaler.scale_):
        assert attribute.shape == (2,)
        assert attribute.dtype == np.float64


def test_transform_returns_new_float64_array_with_expected_values() -> None:
    scaler = StandardScaler().fit([[1, 10], [3, 14]])
    original = np.array([[1, 10], [3, 14]], dtype=np.int64)
    transformed = scaler.transform(original)
    np.testing.assert_array_equal(transformed, [[-1, -1], [1, 1]])
    assert transformed.shape == original.shape
    assert transformed.dtype == np.float64
    assert not np.shares_memory(transformed, original)


def test_fit_transform_matches_separate_calls_and_leaves_scaler_fitted() -> None:
    data = np.array([[2.0, 5.0], [4.0, 8.0], [9.0, 14.0]])
    combined = StandardScaler()
    result = combined.fit_transform(data)
    separate = StandardScaler().fit(data)
    np.testing.assert_allclose(result, separate.transform(data))
    np.testing.assert_array_equal(combined.mean_, separate.mean_)
    np.testing.assert_array_equal(combined.var_, separate.var_)
    np.testing.assert_array_equal(combined.scale_, separate.scale_)
    assert combined.n_features_in_ == 2


def test_successful_refit_replaces_all_statistics() -> None:
    scaler = StandardScaler().fit([[1, 10], [3, 14]])
    scaler.fit([[4], [8], [12]])
    np.testing.assert_array_equal(scaler.mean_, [8])
    np.testing.assert_allclose(scaler.var_, [32 / 3])
    np.testing.assert_allclose(scaler.scale_, [np.sqrt(32 / 3)])
    assert scaler.n_features_in_ == 1
    assert scaler.transform([[8]]).shape == (1, 1)


def test_fit_and_transform_do_not_mutate_inputs_or_statistics() -> None:
    training = np.array([[1.0, 3.0], [5.0, 3.0]])
    training_before = training.copy()
    scaler = StandardScaler().fit(training)
    np.testing.assert_array_equal(training, training_before)
    learned = tuple(value.copy() for value in (scaler.mean_, scaler.var_, scaler.scale_))
    future = np.array([[9.0, 6.0]])
    future_before = future.copy()
    scaler.transform(future)
    np.testing.assert_array_equal(future, future_before)
    for actual, expected in zip((scaler.mean_, scaler.var_, scaler.scale_), learned, strict=True):
        np.testing.assert_array_equal(actual, expected)


def test_learned_attributes_do_not_exist_before_successful_fit() -> None:
    scaler = StandardScaler()
    for name in ("mean_", "var_", "scale_", "n_features_in_"):
        assert not hasattr(scaler, name)
    with pytest.raises(ValueError):
        scaler.fit([[float("nan")]])
    for name in ("mean_", "var_", "scale_", "n_features_in_"):
        assert not hasattr(scaler, name)


def test_rejects_1d_input() -> None:
    with pytest.raises(ValueError):
        StandardScaler().fit([1, 2, 3])


def test_rejects_zero_samples() -> None:
    with pytest.raises(ValueError):
        StandardScaler().fit(np.empty((0, 2)))


def test_rejects_zero_features() -> None:
    with pytest.raises(ValueError):
        StandardScaler().fit(np.empty((2, 0)))


def test_rejects_ragged_input() -> None:
    with pytest.raises(ValueError):
        StandardScaler().fit([[1, 2], [3]])


def test_rejects_boolean_input() -> None:
    with pytest.raises(ValueError):
        StandardScaler().fit([[True], [False]])


def test_rejects_object_input() -> None:
    with pytest.raises(ValueError):
        StandardScaler().fit(np.array([[1.0]], dtype=object))


def test_rejects_string_input() -> None:
    with pytest.raises(ValueError):
        StandardScaler().fit([["1"], ["2"]])


def test_rejects_complex_input() -> None:
    with pytest.raises(ValueError):
        StandardScaler().fit([[1 + 2j]])


def test_rejects_sparse_input() -> None:
    class SparseInput:
        def tocoo(self) -> None:
            pass

        def toarray(self) -> np.ndarray:
            return np.array([[1.0]])

    with pytest.raises(ValueError):
        StandardScaler().fit(SparseInput())  # type: ignore


def test_rejects_nan_input() -> None:
    with pytest.raises(ValueError):
        StandardScaler().fit([[float("nan")]])


def test_rejects_infinite_input() -> None:
    with pytest.raises(ValueError):
        StandardScaler().fit([[float("inf")]])


def test_transform_checks_fitted_state_before_input() -> None:
    with pytest.raises(RuntimeError):
        StandardScaler().transform([[float("nan")]])


def test_transform_rejects_invalid_input() -> None:
    scaler = StandardScaler().fit([[1.0]])
    with pytest.raises(ValueError):
        scaler.transform([[float("nan")]])


def test_transform_rejects_wrong_feature_count() -> None:
    scaler = StandardScaler().fit([[1, 2]])
    with pytest.raises(ValueError, match="Expected 2 features"):
        scaler.transform([[1]])


def test_single_sample_uses_unit_scales_and_transforms_to_zero() -> None:
    scaler = StandardScaler()
    transformed = scaler.fit_transform([[2, 7]])
    np.testing.assert_array_equal(scaler.var_, [0, 0])
    np.testing.assert_array_equal(scaler.scale_, [1, 1])
    np.testing.assert_array_equal(transformed, [[0, 0]])


def test_one_feature_preserves_two_dimensional_shape() -> None:
    result = StandardScaler().fit_transform([[1], [3]])
    assert result.shape == (2, 1)
    np.testing.assert_array_equal(result, [[-1], [1]])


def test_constant_feature_preserves_future_deviation() -> None:
    scaler = StandardScaler().fit([[2, 5], [4, 5]])
    np.testing.assert_array_equal(scaler.var_, [1, 0])
    np.testing.assert_array_equal(scaler.scale_, [1, 1])
    np.testing.assert_array_equal(scaler.transform([[2, 5], [4, 5]]), [[-1, 0], [1, 0]])
    np.testing.assert_array_equal(scaler.transform([[3, 8]]), [[0, 3]])


def test_failed_refit_preserves_previous_fitted_state() -> None:
    scaler = StandardScaler().fit([[1, 10], [3, 14]])
    learned = tuple(value.copy() for value in (scaler.mean_, scaler.var_, scaler.scale_))
    with pytest.raises(ValueError):
        scaler.fit([[float("nan")]])
    for actual, expected in zip((scaler.mean_, scaler.var_, scaler.scale_), learned, strict=True):
        np.testing.assert_array_equal(actual, expected)
    assert scaler.n_features_in_ == 2
    np.testing.assert_array_equal(scaler.transform([[1, 10]]), [[-1, -1]])


def test_failed_numerical_refit_preserves_previous_fitted_state() -> None:
    scaler = StandardScaler().fit([[1.0], [3.0]])
    learned = tuple(value.copy() for value in (scaler.mean_, scaler.var_, scaler.scale_))
    with pytest.raises(ValueError, match="non-finite scaling statistics"):
        scaler.fit([[1e308], [-1e308]])
    for actual, expected in zip((scaler.mean_, scaler.var_, scaler.scale_), learned, strict=True):
        np.testing.assert_array_equal(actual, expected)
    assert scaler.n_features_in_ == 1


def test_transform_rejects_nonfinite_computed_values() -> None:
    scaler = StandardScaler().fit([[1e308]])
    with pytest.raises(ValueError, match="transformed to finite values"):
        scaler.transform([[-1e308]])
