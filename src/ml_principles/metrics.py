from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]


def _is_sparse_like(value: object) -> bool:
    """Return whether an object exposes a sparse-array interface."""
    return callable(getattr(value, "toarray", None)) and callable(getattr(value, "tocsr", None))


def _validate_target(
    values: ArrayLike,
    *,
    name: str,
) -> FloatArray:
    """Validate one regression target and return a float64 copy."""
    if _is_sparse_like(values):
        raise ValueError(f"{name} must not be sparse.")

    try:
        array = np.asarray(values)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a non-ragged numeric one-dimensional array.") from exc

    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional.")

    if array.size == 0:
        raise ValueError(f"{name} must not be empty.")

    if np.issubdtype(array.dtype, np.bool_):
        raise ValueError(f"{name} must not contain Boolean values.")

    if np.issubdtype(array.dtype, np.complexfloating):
        raise ValueError(f"{name} must not contain complex values.")

    if array.dtype.kind not in {"i", "u", "f"}:
        raise ValueError(f"{name} must contain real numeric values.")

    try:
        result = np.array(
            array,
            dtype=np.float64,
            copy=True,
        )
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be convertible to float64.") from exc

    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite values.")

    return result


def _validate_targets(
    y_true: ArrayLike,
    y_pred: ArrayLike,
) -> tuple[FloatArray, FloatArray]:
    """Validate matching regression targets."""
    true = _validate_target(y_true, name="y_true")
    pred = _validate_target(y_pred, name="y_pred")

    if true.shape != pred.shape:
        raise ValueError("y_true and y_pred must have matching lengths.")

    return true, pred


def _require_finite_result(
    value: np.float64,
    *,
    name: str,
) -> float:
    """Convert a finite NumPy scalar to a Python float."""
    if not np.isfinite(value):
        raise ValueError(f"{name} could not produce a finite result.")

    return float(value)


def mean_squared_error(
    y_true: ArrayLike,
    y_pred: ArrayLike,
) -> float:
    """Return the mean squared error between two target arrays.

    Accepts matching, nonempty, finite, one-dimensional Python numeric
    sequences or NumPy arrays containing real integer or floating-point
    values. Inputs are converted to float64 for calculation.

    Returns:
        The mean of ``(y_pred - y_true) ** 2`` as a Python float.

    Raises:
        ValueError: If either input is scalar, multidimensional, empty,
            Boolean, string, object, complex, sparse, ragged, nonfinite,
            or if their lengths differ. Also raised when finite inputs
            overflow or otherwise cannot produce a finite result.

    The input objects are not mutated.
    """
    true, pred = _validate_targets(y_true, y_pred)

    try:
        with np.errstate(
            over="raise",
            invalid="raise",
            divide="raise",
        ):
            residuals = pred - true
            result = np.mean(
                residuals**2,
                dtype=np.float64,
            )
    except FloatingPointError as exc:
        raise ValueError("mean_squared_error could not produce a finite result.") from exc

    return _require_finite_result(
        result,
        name="mean_squared_error",
    )


def mean_absolute_error(
    y_true: ArrayLike,
    y_pred: ArrayLike,
) -> float:
    """Return the mean absolute error between two target arrays.

    Accepts matching, nonempty, finite, one-dimensional Python numeric
    sequences or NumPy arrays containing real integer or floating-point
    values. Inputs are converted to float64 for calculation.

    Returns:
        The mean absolute difference between ``y_true`` and ``y_pred``
        as a Python float.

    Raises:
        ValueError: If either input is scalar, multidimensional, empty,
            Boolean, string, object, complex, sparse, ragged, nonfinite,
            or if their lengths differ. Also raised when finite inputs
            overflow or otherwise cannot produce a finite result.

    The input objects are not mutated.
    """
    true, pred = _validate_targets(y_true, y_pred)

    try:
        with np.errstate(
            over="raise",
            invalid="raise",
            divide="raise",
        ):
            residuals = pred - true
            result = np.mean(
                np.abs(residuals),
                dtype=np.float64,
            )
    except FloatingPointError as exc:
        raise ValueError("mean_absolute_error could not produce a finite result.") from exc

    return _require_finite_result(
        result,
        name="mean_absolute_error",
    )


def r2_score(
    y_true: ArrayLike,
    y_pred: ArrayLike,
) -> float:
    """Return the coefficient of determination R².

    Accepts matching, finite, one-dimensional Python numeric sequences
    or NumPy arrays containing real integer or floating-point values.
    Inputs are converted to float64 for calculation.

    R² is calculated as ``1 - SS_res / SS_tot``, where ``SS_res`` is
    the residual sum of squares and ``SS_tot`` is the total sum of
    squares around the mean of ``y_true``.

    Returns:
        R² as a Python float. The value may be negative when predictions
        perform worse than the target-mean reference prediction.

    Raises:
        ValueError: If either input is scalar, multidimensional, empty,
            Boolean, string, object, complex, sparse, ragged, nonfinite,
            or if their lengths differ. Also raised when fewer than two
            targets are supplied, when ``y_true`` is constant, or when
            finite inputs overflow or cannot produce a finite result.

    The input objects are not mutated.
    """
    true, pred = _validate_targets(y_true, y_pred)

    if true.size < 2:
        raise ValueError("R² requires at least two targets.")

    try:
        with np.errstate(
            over="raise",
            invalid="raise",
            divide="raise",
        ):
            true_mean = np.mean(
                true,
                dtype=np.float64,
            )

            centered = true - true_mean
            ss_tot = np.sum(
                centered**2,
                dtype=np.float64,
            )

            if ss_tot == 0.0:
                raise ValueError("R² is undefined when y_true is constant.")

            residuals = true - pred
            ss_res = np.sum(
                residuals**2,
                dtype=np.float64,
            )

            result = np.float64(1.0) - ss_res / ss_tot
    except FloatingPointError as exc:
        raise ValueError("r2_score could not produce a finite result.") from exc

    return _require_finite_result(
        result,
        name="r2_score",
    )


def mean_baseline(
    y_train: ArrayLike,
    n_predictions: int,
) -> FloatArray:
    """Create constant predictions using only the training-target mean.

    Accepts a nonempty, finite, one-dimensional Python numeric sequence
    or NumPy array containing real integer or floating-point training
    targets. ``n_predictions`` must be a positive integer; Boolean
    values are not accepted as integers.

    The prediction value is learned exclusively from the mean of
    ``y_train``. Evaluation or test targets must never be used when
    constructing this baseline.

    Returns:
        A one-dimensional float64 NumPy array of length
        ``n_predictions`` whose elements equal the training-target mean.

    Raises:
        ValueError: If ``y_train`` is scalar, multidimensional, empty,
            Boolean, string, object, complex, sparse, ragged, or
            nonfinite; if ``n_predictions`` is not a positive integer;
            or if calculating the training mean overflows or otherwise
            cannot produce a finite value.

    The training input is not mutated.
    """
    train = _validate_target(
        y_train,
        name="y_train",
    )

    if isinstance(
        n_predictions,
        (bool, np.bool_),
    ) or not isinstance(
        n_predictions,
        (int, np.integer),
    ):
        raise ValueError("n_predictions must be a positive integer.")

    count = int(n_predictions)

    if count <= 0:
        raise ValueError("n_predictions must be a positive integer.")

    try:
        with np.errstate(
            over="raise",
            invalid="raise",
            divide="raise",
        ):
            train_mean = np.mean(
                train,
                dtype=np.float64,
            )
    except FloatingPointError as exc:
        raise ValueError("Training-target mean could not produce a finite value.") from exc

    if not np.isfinite(train_mean):
        raise ValueError("Training-target mean could not produce a finite value.")

    return np.full(
        count,
        train_mean,
        dtype=np.float64,
    )


__all__ = [
    "mean_squared_error",
    "mean_absolute_error",
    "r2_score",
    "mean_baseline",
]
