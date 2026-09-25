"""Behavioral contract for linear regression trained with batch gradient descent."""

import numpy as np
import pytest

from ml_principles.linear_models import LinearRegression


def test_one_update_matches_hand_calculated_full_mse_step() -> None:
    model = LinearRegression(learning_rate=0.1, max_iter=1, tol=0.0)

    assert model.fit([[1], [2]], [3, 5]) is model

    np.testing.assert_allclose(model.coef_, [1.3])
    assert model.intercept_ == 0.8
    np.testing.assert_allclose(model.loss_history_, [17.0, 1.685])
    assert model.n_features_in_ == 1
    assert model.n_iter_ == 1
    assert model.converged_ is False
    np.testing.assert_allclose(model.predict([[1], [2]]), [2.1, 3.4])


def test_known_affine_problem_converges_to_expected_parameters() -> None:
    model = LinearRegression(learning_rate=0.1, max_iter=1000, tol=1e-10)

    model.fit([[-1], [0], [1]], [1, 3, 5])

    np.testing.assert_allclose(model.coef_, [2.0], atol=1e-9)
    assert model.intercept_ == pytest.approx(3.0, abs=1e-9)
    np.testing.assert_allclose(model.predict([[-2], [2]]), [-1, 7], atol=1e-9)
    assert model.converged_ is True
    assert 0 < model.n_iter_ < model.max_iter
    assert model.loss_history_.shape == (model.n_iter_ + 1,)


def test_initially_optimal_model_converges_without_an_update() -> None:
    model = LinearRegression(max_iter=5, tol=0.0).fit([[1], [2]], [0, 0])

    np.testing.assert_array_equal(model.coef_, [0.0])
    assert model.intercept_ == 0.0
    assert model.n_iter_ == 0
    assert model.converged_ is True
    np.testing.assert_array_equal(model.loss_history_, [0.0])


def test_fit_publishes_documented_state_types() -> None:
    model = LinearRegression(max_iter=1, tol=0.0).fit([[1, 2], [3, 4]], [1, 2])

    assert model.coef_.shape == (2,)
    assert model.coef_.dtype == np.float64
    assert isinstance(model.intercept_, float)
    assert isinstance(model.n_features_in_, int)
    assert isinstance(model.n_iter_, int)
    assert model.loss_history_.dtype == np.float64
    assert isinstance(model.converged_, bool)


def test_predict_requires_a_fitted_model() -> None:
    with pytest.raises(RuntimeError, match="fitted"):
        LinearRegression().predict([[1.0]])


@pytest.mark.parametrize(
    "kwargs",
    [
        {"learning_rate": 0.0},
        {"learning_rate": -0.1},
        {"learning_rate": float("inf")},
        {"learning_rate": True},
        {"learning_rate": "0.1"},
        {"max_iter": 0},
        {"max_iter": -1},
        {"max_iter": 1.5},
        {"max_iter": True},
        {"tol": -1.0},
        {"tol": float("nan")},
        {"tol": float("inf")},
        {"tol": True},
    ],
)
def test_fit_rejects_invalid_hyperparameters(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        LinearRegression(**kwargs).fit([[1.0]], [1.0])  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("X", "y"),
    [
        ([1, 2], [1, 2]),
        (np.empty((0, 1)), np.empty(0)),
        (np.empty((2, 0)), [1, 2]),
        ([[1], [2]], [[1], [2]]),
        ([[1], [2]], [1]),
        ([[1], [2]], []),
        ([[True], [False]], [1, 2]),
        ([[1], [2]], [True, False]),
        ([[1 + 1j]], [1]),
        ([[1]], [1 + 1j]),
        ([["1"]], [1]),
        ([[1]], ["1"]),
        (np.array([[1]], dtype=object), [1]),
        ([[1]], np.array([1], dtype=object)),
        ([[float("nan")]], [1]),
        ([[1]], [float("inf")]),
        ([[1, 2], [3]], [1, 2]),
    ],
)
def test_fit_rejects_invalid_training_data(X: object, y: object) -> None:
    with pytest.raises(ValueError):
        LinearRegression().fit(X, y)  # type: ignore[arg-type]


def test_fit_rejects_sparse_like_inputs() -> None:
    class SparseInput:
        def toarray(self) -> np.ndarray:
            return np.array([[1.0]])

        def tocsr(self) -> "SparseInput":
            return self

    with pytest.raises(ValueError, match="sparse"):
        LinearRegression().fit(SparseInput(), [1])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="sparse"):
        LinearRegression().fit([[1]], SparseInput())  # type: ignore[arg-type]


def test_predict_validates_shape_values_and_feature_count() -> None:
    model = LinearRegression(max_iter=1).fit([[1, 2], [3, 4]], [1, 2])

    with pytest.raises(ValueError):
        model.predict([1, 2])
    with pytest.raises(ValueError, match="Expected 2 features"):
        model.predict([[1]])
    with pytest.raises(ValueError):
        model.predict([[float("nan"), 1]])


def test_fit_and_predict_do_not_mutate_caller_arrays() -> None:
    X = np.array([[1.0], [2.0]])
    y = np.array([3.0, 5.0])
    X_before = X.copy()
    y_before = y.copy()
    model = LinearRegression(max_iter=1).fit(X, y)
    future = np.array([[3.0]])
    future_before = future.copy()

    predictions = model.predict(future)

    np.testing.assert_array_equal(X, X_before)
    np.testing.assert_array_equal(y, y_before)
    np.testing.assert_array_equal(future, future_before)
    assert predictions.dtype == np.float64
    assert predictions.shape == (1,)
    assert not np.shares_memory(predictions, future)
