
import numpy as np

from src.data.dataset import GlucoseSeries


def zscore_stats(series: GlucoseSeries) -> tuple[np.ndarray, np.ndarray]:
    """Per-feature mean/std on the full series (use training-only series for Ohio train→test generalization)."""
    v = series.values.astype(np.float64)
    mean = v.mean(axis=0, keepdims=True)
    std = v.std(axis=0, keepdims=True) + 1e-6
    return mean, std


def zscore_stats_segments(segments: list[GlucoseSeries]) -> tuple[np.ndarray, np.ndarray]:
    """Per-feature mean/std over vertically stacked rows of all segments (train portions only)."""
    if not segments:
        raise ValueError("no segments for z-score stats")
    v = np.concatenate([s.values.astype(np.float64) for s in segments], axis=0)
    mean = v.mean(axis=0, keepdims=True)
    std = v.std(axis=0, keepdims=True) + 1e-6
    return mean, std


def build_arrays(
    series: GlucoseSeries,
    lookback: int,
    horizon: int,
    normalize: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    v = series.values.astype(np.float64)
    T, F = v.shape
    if normalize:
        mean = v.mean(axis=0, keepdims=True)
        std = v.std(axis=0, keepdims=True) + 1e-6
        v = (v - mean) / std
    else:
        mean = std = None
    X, y = [], []
    for t in range(lookback, T - horizon):
        X.append(v[t - lookback : t])
        y.append(v[t + horizon, 0])
    return np.stack(X), np.array(y), (mean, std) if normalize else None


def build_arrays_with_stats(
    series: GlucoseSeries,
    lookback: int,
    horizon: int,
    mean: np.ndarray,
    std: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Build (X, y) with fixed z-score stats (e.g. train mean/std applied to official Testing XML)."""
    v = series.values.astype(np.float64)
    v = (v - mean) / std
    T, _F = v.shape
    X, y = [], []
    for t in range(lookback, T - horizon):
        X.append(v[t - lookback : t])
        y.append(v[t + horizon, 0])
    if not X:
        raise ValueError(
            f"Series too short for lookback={lookback}, horizon={horizon}: need T > {lookback + horizon}"
        )
    return np.stack(X), np.array(y)


def build_arrays_with_stats_segments(
    segments: list[GlucoseSeries],
    lookback: int,
    horizon: int,
    mean: np.ndarray,
    std: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Stack sliding-window (X, y) from each segment; no window crosses segment boundaries."""
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    for s in segments:
        if s.n_steps <= lookback + horizon:
            continue
        x_i, y_i = build_arrays_with_stats(s, lookback, horizon, mean, std)
        xs.append(x_i)
        ys.append(y_i)
    if not xs:
        raise ValueError(
            f"No segment long enough for lookback={lookback}, horizon={horizon} "
            f"(need T > {lookback + horizon} per segment)"
        )
    return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0)
