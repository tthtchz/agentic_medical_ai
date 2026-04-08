"""Sliding-window (X, y) tensors and z-score stats for LSTM training on ``GlucoseSeries``."""

import numpy as np

from .dataset import GlucoseSeries

_EPS = 1e-6


def zscore_stats_segments(segments: list[GlucoseSeries]) -> tuple[np.ndarray, np.ndarray]:
    """
    Per-feature mean and standard deviation over all rows of ``segments``.

    Use training-only segments so validation/test stay normalized with train statistics.
    """
    if not segments:
        raise ValueError("no segments for z-score stats")
    v = np.concatenate([s.values.astype(np.float64) for s in segments], axis=0)
    mean = v.mean(axis=0, keepdims=True)
    std = v.std(axis=0, keepdims=True) + _EPS
    return mean, std


def _sliding_xy_normalized(
    values: np.ndarray,
    lookback: int,
    horizon: int,
    mean: np.ndarray,
    std: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply fixed z-score, then for each valid ``t``:

    - ``X``: rows ``t - lookback .. t - 1`` (shape ``(lookback, F)``),
    - ``y``: glucose at ``t + horizon`` (column 0), scalar per window.
    """
    v = (values.astype(np.float64) - mean) / std
    t_max, n_feat = v.shape
    if t_max <= lookback + horizon:
        return np.empty((0, lookback, n_feat)), np.empty((0,))
    xs: list[np.ndarray] = []
    ys: list[float] = []
    for t in range(lookback, t_max - horizon):
        xs.append(v[t - lookback : t])
        ys.append(float(v[t + horizon, 0]))
    return np.stack(xs, axis=0), np.asarray(ys, dtype=np.float64)


def build_arrays_with_stats_segments(
    segments: list[GlucoseSeries],
    lookback: int,
    horizon: int,
    mean: np.ndarray,
    std: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build sliding-window batches from each segment and concatenate.

    No window crosses segment boundaries. Segments shorter than ``lookback + horizon + 1``
    steps are skipped.
    """
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    min_steps = lookback + horizon + 1
    for s in segments:
        if s.n_steps < min_steps:
            continue
        x_i, y_i = _sliding_xy_normalized(
            s.values, lookback, horizon, mean, std
        )
        if x_i.shape[0] == 0:
            continue
        xs.append(x_i)
        ys.append(y_i)
    if not xs:
        raise ValueError(
            f"No segment long enough for lookback={lookback}, horizon={horizon} "
            f"(need at least T >= {min_steps} per segment)"
        )
    return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0)
