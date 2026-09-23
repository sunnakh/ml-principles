"""Reusable preprocessing transforms for dense numeric data."""

from typing import Self

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _validate_input(X: ArrayLike) -> NDArray[np.float64]:
    """Return finite, two-dimensional float64 data without changing the caller's input."""
    if hasattr(X, "tocoo") and hasattr(X, "toarray"):
        raise ValueError("Sparse input is not supported")

    try:
        array = np.asarray(X)
    except (TypeError, ValueError) as exc:
        raise ValueError("X must be a rectangular numeric array") from exc

    if array.ndim != 2 or 0 in array.shape:
        raise ValueError("X must have at least one row and one column")
    if array.dtype.kind not in "iuf":
        raise ValueError("X must contain real integer or floating-point values")

    try:
        result = np.asarray(array, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("X cannot be converted to float64") from exc
    if not np.all(np.isfinite(result)):
        raise ValueError("X must contain only finite values")
    return result


class StandardScaler:
    """Center features and divide by their training population standard deviation."""

    mean_: NDArray[np.float64]
    var_: NDArray[np.float64]
    scale_: NDArray[np.float64]
    n_features_in_: int

    def fit(self, X: ArrayLike) -> Self:
        """Learn per-feature statistics from a dense feature matrix.

        Args:
            X: Finite real values with shape ``(n_samples, n_features)``.

        Returns:
            This fitted scaler.

        Raises:
            ValueError: If ``X`` is unsupported or finite statistics cannot be computed.
        """
        data = _validate_input(X)
        try:
            with np.errstate(over="raise", invalid="raise"):
                mean = np.mean(data, axis=0, dtype=np.float64)
                variance = np.var(data, axis=0, ddof=0, dtype=np.float64)
                scale = np.sqrt(variance)
        except FloatingPointError as exc:
            raise ValueError("X produces non-finite scaling statistics") from exc

        if not all(np.all(np.isfinite(value)) for value in (mean, variance, scale)):
            raise ValueError("X produces non-finite scaling statistics")
        scale[scale == 0.0] = 1.0

        # Publish statistics only after input validation and all computations succeed.
        self.mean_ = mean
        self.var_ = variance
        self.scale_ = scale
        self.n_features_in_ = data.shape[1]
        return self

    def transform(self, X: ArrayLike) -> NDArray[np.float64]:
        """Standardize a dense feature matrix using the learned statistics.

        Args:
            X: Finite real values with shape ``(n_samples, n_features_in_)``.

        Returns:
            A new float64 array with the same shape as ``X``.

        Raises:
            RuntimeError: If the scaler has not been fitted.
            ValueError: If ``X`` is unsupported, has the wrong feature count, or cannot
                be transformed to finite values.
        """
        if not hasattr(self, "n_features_in_"):
            raise RuntimeError("StandardScaler must be fitted before transform")

        data = _validate_input(X)
        if data.shape[1] != self.n_features_in_:
            raise ValueError(f"Expected {self.n_features_in_} features, got {data.shape[1]}")

        try:
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                transformed = (data - self.mean_) / self.scale_
        except FloatingPointError as exc:
            raise ValueError("X cannot be transformed to finite values") from exc

        if not np.all(np.isfinite(transformed)):
            raise ValueError("X cannot be transformed to finite values")
        return transformed

    def fit_transform(self, X: ArrayLike) -> NDArray[np.float64]:
        """Learn feature statistics and standardize the same feature matrix.

        Args:
            X: Finite real values with shape ``(n_samples, n_features)``.

        Returns:
            A new float64 array with the same shape as ``X``.

        Raises:
            ValueError: If ``X`` is unsupported or finite results cannot be computed.
        """
        return self.fit(X).transform(X)
