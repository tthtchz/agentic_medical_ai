"""OhioT1DM *-ws-*.xml loaders and train/validation time splits for multivariate CGM-style series."""

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------

_OHIO_TS_FMT = "%d-%m-%Y %H:%M:%S"
_DEFAULT_GRID_MINUTES = 5.0
_DEFAULT_CGM_GAP_MINUTES = 60.0


def _parse_ohio_ts(raw: str) -> datetime:
    return datetime.strptime(raw.strip(), _OHIO_TS_FMT)


# -----------------------------------------------------------------------------
# Core type
# -----------------------------------------------------------------------------


@dataclass
class GlucoseSeries:
    """Multivariate series on a uniform time grid (e.g. 5-minute steps)."""

    values: np.ndarray  # (T, F): glucose, insulin, carbs, ...
    index_minutes: np.ndarray  # (T,) monotonic, origin 0 within the series

    @property
    def glucose(self) -> np.ndarray:
        return self.values[:, 0]

    @property
    def n_steps(self) -> int:
        return int(self.values.shape[0])

    @property
    def n_features(self) -> int:
        return int(self.values.shape[1])


def concat_glucose_series(
    parts: list[GlucoseSeries],
    gap_minutes: float = _DEFAULT_GRID_MINUTES,
) -> GlucoseSeries:
    """Concatenate series in order; inserts a fake gap of ``gap_minutes`` between segments."""
    if not parts:
        raise ValueError("empty series list")
    vals = np.concatenate([p.values for p in parts], axis=0)
    chunks: list[np.ndarray] = []
    acc = 0.0
    for p in parts:
        chunks.append(p.index_minutes + acc)
        acc = float(chunks[-1][-1] + gap_minutes)
    index = np.concatenate(chunks)
    return GlucoseSeries(values=vals, index_minutes=index)


def time_split_series(
    series: GlucoseSeries,
    train_frac: float = 0.8,
    min_tail_steps: int = 200,
) -> tuple[GlucoseSeries, GlucoseSeries]:
    """First ``train_frac`` of steps vs remainder; for Ohio val, use ``min_tail_steps=0`` per segment."""
    T = series.n_steps
    n_tr = int(T * train_frac)
    n_tr = max(n_tr, 1)
    n_tr = min(n_tr, T - 1)
    if min_tail_steps > 0 and T - n_tr < min_tail_steps:
        n_tr = T - min_tail_steps
    if n_tr < 1:
        raise ValueError(f"Series too short for time split (T={T})")
    v = series.values
    im = series.index_minutes
    return (
        GlucoseSeries(values=v[:n_tr].copy(), index_minutes=im[:n_tr].copy()),
        GlucoseSeries(values=v[n_tr:].copy(), index_minutes=im[n_tr:].copy()),
    )


def time_split_segments(
    segments: list[GlucoseSeries],
    train_frac: float,
    min_tail_steps: int = 0,
) -> tuple[list[GlucoseSeries], list[GlucoseSeries]]:
    """Apply :func:`time_split_series` to each segment; skip segments that cannot be split."""
    train_parts: list[GlucoseSeries] = []
    val_parts: list[GlucoseSeries] = []
    for seg in segments:
        if seg.n_steps < 2:
            continue
        try:
            tr, va = time_split_series(
                seg, train_frac=train_frac, min_tail_steps=min_tail_steps
            )
        except ValueError:
            continue
        train_parts.append(tr)
        val_parts.append(va)
    if not train_parts:
        raise ValueError("No segment long enough for train/validation time split")
    return train_parts, val_parts


# -----------------------------------------------------------------------------
# Ohio ws XML: segment grid + insulin / carbs
# -----------------------------------------------------------------------------


def _basal_rate_u_per_h(t: datetime, basal: list[tuple[datetime, float]]) -> float:
    rate = basal[0][1]
    for t_change, r in basal:
        if t_change <= t:
            rate = r
    return rate


def _temp_rate_u_per_h(
    t: datetime, intervals: list[tuple[datetime, datetime, float]]
) -> float | None:
    for a, b, rate in intervals:
        if a <= t < b:
            return rate
    return None


def _split_cgm_pairs_on_gap(
    cgm_pairs_sorted: list[tuple[datetime, float]],
    gap_minutes: float,
) -> list[list[tuple[datetime, float]]]:
    """Split sorted CGM pairs when consecutive samples are more than ``gap_minutes`` apart."""
    if not cgm_pairs_sorted:
        return []
    segs: list[list[tuple[datetime, float]]] = [[cgm_pairs_sorted[0]]]
    for i in range(1, len(cgm_pairs_sorted)):
        dt_min = (
            cgm_pairs_sorted[i][0] - cgm_pairs_sorted[i - 1][0]
        ).total_seconds() / 60.0
        if dt_min > gap_minutes:
            segs.append([cgm_pairs_sorted[i]])
        else:
            segs[-1].append(cgm_pairs_sorted[i])
    return segs


def _build_ohio_segment_grid(
    cgm_segment: list[tuple[datetime, float]],
    t0_dt: datetime,
    t_end_dt: datetime,
    basal: list[tuple[datetime, float]],
    temp_intervals: list[tuple[datetime, datetime, float]],
    bolus_doses: list[tuple[datetime, float]],
    meal_carbs: list[tuple[datetime, float]],
    grid_minutes: float,
) -> GlucoseSeries:
    """One contiguous CGM segment on a 5-min grid; glucose interpolated only within the segment."""

    def minutes_from_t0(dt: datetime) -> float:
        return (dt - t0_dt).total_seconds() / 60.0

    t_end = minutes_from_t0(t_end_dt)
    n = int(np.floor(t_end / grid_minutes)) + 1
    if n < 1:
        raise ValueError("empty segment grid")
    grid_m = np.arange(n, dtype=np.float64) * grid_minutes

    g = np.full(n, np.nan, dtype=np.float64)
    for dt, val in cgm_segment:
        idx = int(round(minutes_from_t0(dt) / grid_minutes))
        if 0 <= idx < n:
            g[idx] = val
    g = pd.Series(g).interpolate(limit_direction="both").to_numpy()

    ins = np.zeros(n, dtype=np.float64)
    five_over_sixty = grid_minutes / 60.0
    for k in range(n):
        tdt = t0_dt + pd.Timedelta(minutes=float(grid_m[k]))
        r_temp = _temp_rate_u_per_h(tdt, temp_intervals)
        r_base = _basal_rate_u_per_h(tdt, basal)
        rate = r_temp if r_temp is not None else r_base
        ins[k] += rate * five_over_sixty
    for dt_b, dose in bolus_doses:
        idx = int(round(minutes_from_t0(dt_b) / grid_minutes))
        if 0 <= idx < n:
            ins[idx] += dose

    carbs = np.zeros(n, dtype=np.float64)
    for dt_m, carb in meal_carbs:
        idx = int(round(minutes_from_t0(dt_m) / grid_minutes))
        if 0 <= idx < n:
            carbs[idx] += carb

    vals = np.stack([g, ins, carbs], axis=1)
    return GlucoseSeries(values=vals, index_minutes=grid_m)


def load_ohio_ws_xml_segments(
    path: Path,
    grid_minutes: float = _DEFAULT_GRID_MINUTES,
    gap_minutes: float = _DEFAULT_CGM_GAP_MINUTES,
) -> list[GlucoseSeries]:
    """
    Parse Ohio *-ws-*.xml into contiguous CGM segments.

    Splits when consecutive CGM samples are more than ``gap_minutes`` apart. Each segment has a
    5-minute grid from first to last sample; glucose is interpolated only inside the segment.
    """
    tree = ET.parse(path)
    root = tree.getroot()
    g_parent = root.find("glucose_level")
    if g_parent is None:
        raise ValueError(f"No glucose_level in {path}")

    cgm_pairs: list[tuple[datetime, float]] = []
    for ev in g_parent.findall("event"):
        ts_raw = ev.attrib.get("ts")
        val_raw = ev.attrib.get("value")
        if not ts_raw or val_raw is None:
            continue
        cgm_pairs.append((_parse_ohio_ts(ts_raw), float(val_raw)))
    if len(cgm_pairs) < 10:
        raise ValueError(f"No usable CGM events in {path}")

    cgm_pairs.sort(key=lambda x: x[0])
    seg_pairs = _split_cgm_pairs_on_gap(cgm_pairs, gap_minutes)

    basal: list[tuple[datetime, float]] = []
    b_el = root.find("basal")
    if b_el is not None:
        for ev in b_el.findall("event"):
            tr = ev.attrib.get("ts")
            vr = ev.attrib.get("value")
            if tr and vr is not None:
                basal.append((_parse_ohio_ts(tr), float(vr)))
    basal.sort(key=lambda x: x[0])
    t_anchor = cgm_pairs[0][0]
    if not basal:
        basal = [(t_anchor, 0.0)]

    temp_intervals: list[tuple[datetime, datetime, float]] = []
    tb_el = root.find("temp_basal")
    if tb_el is not None:
        for ev in tb_el.findall("event"):
            a = ev.attrib.get("ts_begin")
            b = ev.attrib.get("ts_end")
            vr = ev.attrib.get("value")
            if a and b and vr is not None:
                temp_intervals.append(
                    (_parse_ohio_ts(a), _parse_ohio_ts(b), float(vr))
                )

    bolus_doses: list[tuple[datetime, float]] = []
    bol_el = root.find("bolus")
    if bol_el is not None:
        for ev in bol_el.findall("event"):
            a = ev.attrib.get("ts_begin")
            d = ev.attrib.get("dose")
            if a and d is not None:
                bolus_doses.append((_parse_ohio_ts(a), float(d)))

    meal_carbs: list[tuple[datetime, float]] = []
    m_el = root.find("meal")
    if m_el is not None:
        for ev in m_el.findall("event"):
            ts_m = ev.attrib.get("ts")
            c = ev.attrib.get("carbs")
            if ts_m and c is not None:
                meal_carbs.append((_parse_ohio_ts(ts_m), float(c)))

    out: list[GlucoseSeries] = []
    for seg in seg_pairs:
        t0_dt = seg[0][0]
        t_end_dt = seg[-1][0]
        out.append(
            _build_ohio_segment_grid(
                seg,
                t0_dt,
                t_end_dt,
                basal,
                temp_intervals,
                bolus_doses,
                meal_carbs,
                grid_minutes,
            )
        )
    return out


# -----------------------------------------------------------------------------
# Ohio multi-subject / testing entry points
# -----------------------------------------------------------------------------


def ohio_subject_ids(training_dir: Path) -> list[str]:
    return sorted(
        {p.name.split("-")[0] for p in Path(training_dir).glob("*-ws-training.xml")}
    )


def load_ohio_training_segments(
    training_dir: Path,
    holdout_subject: str | None = None,
    gap_minutes: float = _DEFAULT_CGM_GAP_MINUTES,
    grid_minutes: float = _DEFAULT_GRID_MINUTES,
) -> list[GlucoseSeries]:
    """
    All ``*-ws-training.xml`` under ``training_dir`` (optional holdout), each file split into
    contiguous CGM segments; flat list, no cross-subject windows.
    """
    td = Path(training_dir)
    xmls = sorted(td.glob("*-ws-training.xml"))
    if not xmls:
        raise ValueError(f"No *-ws-training.xml in {td}")
    if holdout_subject:
        pref = f"{holdout_subject}-"
        xmls = [p for p in xmls if not p.name.startswith(pref)]
    if not xmls:
        raise ValueError("No training files left after holdout exclusion")
    out: list[GlucoseSeries] = []
    for p in xmls:
        segs = load_ohio_ws_xml_segments(
            p, grid_minutes=grid_minutes, gap_minutes=gap_minutes
        )
        out.extend(segs)
    return out


def load_ohio_testing_subject(
    testing_dir: Path,
    subject_id: str,
    gap_minutes: float = _DEFAULT_CGM_GAP_MINUTES,
    grid_minutes: float = _DEFAULT_GRID_MINUTES,
) -> list[GlucoseSeries]:
    """Official Testing split: ``{id}-ws-testing.xml`` as contiguous CGM segments."""
    path = Path(testing_dir) / f"{subject_id}-ws-testing.xml"
    if not path.is_file():
        raise FileNotFoundError(
            f"Expected Testing XML at {path}. Place OhioT1DM Testing export under this directory."
        )
    return load_ohio_ws_xml_segments(path, grid_minutes=grid_minutes, gap_minutes=gap_minutes)
