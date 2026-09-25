"""Linear models implemented with NumPy."""

from numbers import Integral, Real
from typing import Self

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]


def _validate_hyperparameters(
    learning_rate: object,
    max_iter: object,
    tol: object,
) -> tuple[float, int, float]:
    """Validate and normalize gradient-descent hyperparameters."""
    if isinstance(learning_rate, bool) or not isinstance(learning_rate, Real):
        raise ValueError("learning_rate must be a finite real number greater than zero")
    learning_rate_value = float(learning_rate)
    if not np.isfinite(learning_rate_value) or learning_rate_value <= 0.0:
        raise ValueError("learning_rate must be a finite real number greater than zero")

    if isinstance(max_iter, bool) or not isinstance(max_iter, Integral) or max_iter <= 0:
        raise ValueError("max_iter must be a positive integer")

    if isinstance(tol, bool) or not isinstance(tol, Real):
        raise ValueError("tol must be a finite real number greater than or equal to zero")
    tolerance = float(tol)
    if not np.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("tol must be a finite real number greater than or equal to zero")

    return learning_rate_value, int(max_iter), tolerance


def _is_sparse_like(value: object) -> bool:
    """Return whether an object exposes a common sparse-array interface."""
    has_conversion = callable(getattr(value, "toarray", None))
    has_sparse_format = callable(getattr(value, "tocsr", None)) or callable(
        getattr(value, "tocoo", None)
    )
    return has_conversion and has_sparse_format


def _as_numeric_array(value: ArrayLike, *, name: str) -> np.ndarray:
    """Convert an input to an array after rejecting unsupported sparse data."""
    if _is_sparse_like(value):
        raise ValueError(f"{name} must not be sparse")

    try:
        array = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a rectangular numeric array") from exc

    if array.dtype.kind not in "iuf":
        raise ValueError(f"{name} must contain real integer or floating-point values")
    return array


def _validate_features(X: ArrayLike) -> FloatArray:
    """Return a finite two-dimensional float64 copy of a feature matrix."""
    array = _as_numeric_array(X, name="X")
    if array.ndim != 2 or 0 in array.shape:
        raise ValueError("X must have at least one row and one column")

    try:
        result = np.array(array, dtype=np.float64, copy=True)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("X must be convertible to float64") from exc
    if not np.all(np.isfinite(result)):
        raise ValueError("X must contain only finite values")
    return result


def _validate_target(y: ArrayLike) -> FloatArray:
    """Return a finite one-dimensional float64 copy of a target array."""
    array = _as_numeric_array(y, name="y")
    if array.ndim != 1 or array.size == 0:
        raise ValueError("y must be a nonempty one-dimensional array")

    try:
        result = np.array(array, dtype=np.float64, copy=True)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("y must be convertible to float64") from exc
    if not np.all(np.isfinite(result)):
        raise ValueError("y must contain only finite values")
    return result


def _loss_and_gradients(
    X: FloatArray,
    y: FloatArray,
    coefficients: FloatArray,
    intercept: float,
) -> tuple[float, FloatArray, float]:
    """Return full MSE and its gradients at one parameter state."""
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            predictions = X @ coefficients + intercept
            residuals = predictions - y
            loss = np.mean(residuals**2, dtype=np.float64)
            gradient_coef = (2.0 / X.shape[0]) * (X.T @ residuals)
            gradient_intercept = (2.0 / X.shape[0]) * np.sum(
                residuals,
                dtype=np.float64,
            )
    except FloatingPointError as exc:
        raise FloatingPointError("Training produced a nonfinite numerical result") from exc

    if not (
        np.isfinite(loss) and np.all(np.isfinite(gradient_coef)) and np.isfinite(gradient_intercept)
    ):
        raise FloatingPointError("Training produced a nonfinite numerical result")
    return float(loss), np.asarray(gradient_coef, dtype=np.float64), float(gradient_intercept)


class LinearRegression:
    """Fit ordinary linear regression with deterministic batch gradient descent.

    Args:
        learning_rate: Positive finite step size used for every parameter update.
        max_iter: Maximum number of full-batch parameter updates.
        tol: Nonnegative convergence threshold for the combined gradient infinity norm.
    """

    coef_: NDArray[np.float64]
    intercept_: float
    n_features_in_: int
    n_iter_: int
    loss_history_: NDArray[np.float64]
    converged_: bool

    def __init__(
        self,
        *,
        learning_rate: float = 0.01,
        max_iter: int = 1000,
        tol: float = 1e-6,
    ) -> None:
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, X: ArrayLike, y: ArrayLike) -> Self:
        """Fit coefficients and an intercept with batch gradient descent.

        Args:
            X: Finite real feature values with shape ``(n_samples, n_features)``.
            y: Finite real targets with shape ``(n_samples,)``.

        Returns:
            This fitted estimator.

        Raises:
            ValueError: If hyperparameters or training data violate the public contract.
            FloatingPointError: If training produces a nonfinite numerical result.
        """
        learning_rate, max_iter, tolerance = _validate_hyperparameters(
            self.learning_rate,
            self.max_iter,
            self.tol,
        )
        data = _validate_features(X)
        target = _validate_target(y)
        if data.shape[0] != target.shape[0]:
            raise ValueError("X and y must contain the same number of samples")

        coefficients = np.zeros(data.shape[1], dtype=np.float64)
        intercept = 0.0
        losses: list[float] = []
        n_iter = 0
        converged = False

        while True:
            loss, gradient_coef, gradient_intercept = _loss_and_gradients(
                data,
                target,
                coefficients,
                intercept,
            )
            losses.append(loss)

            gradient_max = max(
                float(np.max(np.abs(gradient_coef))),
                abs(gradient_intercept),
            )
            if gradient_max <= tolerance:
                converged = True
                break
            if n_iter == max_iter:
                break

            try:
                with np.errstate(over="raise", invalid="raise"):
                    coefficients = coefficients - learning_rate * gradient_coef
                    intercept = intercept - learning_rate * gradient_intercept
            except FloatingPointError as exc:
                raise FloatingPointError("Training produced nonfinite model parameters") from exc
            if not np.all(np.isfinite(coefficients)) or not np.isfinite(intercept):
                raise FloatingPointError("Training produced nonfinite model parameters")
            n_iter += 1

        self.coef_ = coefficients
        self.intercept_ = intercept
        self.n_features_in_ = data.shape[1]
        self.n_iter_ = n_iter
        self.loss_history_ = np.asarray(losses, dtype=np.float64)
        self.converged_ = converged
        return self

    def predict(self, X: ArrayLike) -> NDArray[np.float64]:
        """Return one float64 prediction for every input row.

        Args:
            X: Finite real feature values with shape
                ``(n_samples, n_features_in_)``.

        Returns:
            Predictions with shape ``(n_samples,)``.

        Raises:
            RuntimeError: If the estimator has not been fitted.
            ValueError: If the input is unsupported or has the wrong feature count.
            FloatingPointError: If prediction produces nonfinite values.
        """
        if not hasattr(self, "coef_"):
            raise RuntimeError("LinearRegression must be fitted before predict")

        data = _validate_features(X)
        if data.shape[1] != self.n_features_in_:
            raise ValueError(f"Expected {self.n_features_in_} features, got {data.shape[1]}")

        try:
            with np.errstate(over="raise", invalid="raise"):
                predictions = data @ self.coef_ + self.intercept_
        except FloatingPointError as exc:
            raise FloatingPointError("Prediction produced nonfinite values") from exc
        if not np.all(np.isfinite(predictions)):
            raise FloatingPointError("Prediction produced nonfinite values")
        return np.asarray(predictions, dtype=np.float64)
