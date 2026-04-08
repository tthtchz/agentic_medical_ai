
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

_OHIO_TS_FMT = "%d-%m-%Y %H:%M:%S"


def _parse_ohio_ts(raw: str) -> datetime:
    return datetime.strptime(raw.strip(), _OHIO_TS_FMT)


@dataclass
class GlucoseSeries:
    """Multivariate CGM-style series aligned on a uniform 5-minute grid."""

    values: np.ndarray  # (T, F): [glucose_mgdl, insulin_u_min, carbs_g_min, ...]
    index_minutes: np.ndarray  # (T,) monotonic from 0

    @property
    def glucose(self) -> np.ndarray:
        return self.values[:, 0]

    @property
    def n_steps(self) -> int:
        return int(self.values.shape[0])

    @property
    def n_features(self) -> int:
        return int(self.values.shape[1])


def time_split_series(
    series: GlucoseSeries,
    train_frac: float = 0.8,
    min_tail_steps: int = 200,
) -> tuple[GlucoseSeries, GlucoseSeries]:
    """
    Split along time (first `train_frac` vs remainder) for train/validation from the same source.

    Used for Ohio: validation must come from Training data, not from the official Testing XML.

    For per-segment splits, pass ``min_tail_steps=0`` so short segments are not rejected.
    """
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
    """Split each segment in time; skip segments too short for a valid split."""
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


def _infer_ts_minutes(elem: ET.Element) -> float | None:
    for key in ("ts", "timestamp", "time", "t"):
        raw = elem.attrib.get(key)
        if raw is None:
            continue
        try:
            return float(raw)
        except ValueError:
            pass
    return None


def _infer_glucose_value(elem: ET.Element) -> float | None:
    for key in ("value", "val", "mgdl", "glucose"):
        raw = elem.attrib.get(key)
        if raw is None:
            continue
        try:
            return float(raw)
        except ValueError:
            pass
    if elem.text and elem.text.strip():
        try:
            return float(elem.text.strip())
        except ValueError:
            pass
    return None


def _collect_glucose_events(root: ET.Element) -> list[tuple[float, float]]:
    """Fallback: heuristic scan (non-Ohio XML). Prefer load_ohio_ws_xml for OhioT1DM."""
    events: list[tuple[float, float]] = []
    interesting = {
        "glucose",
        "cgm",
        "egv",
        "sensor",
        "bg",
        "event",
        "glucose_value",
    }
    for el in root.iter():
        tag = el.tag.split("}")[-1].lower()
        if tag in interesting or "glucose" in tag:
            t = _infer_ts_minutes(el)
            v = _infer_glucose_value(el)
            if t is not None and v is not None:
                events.append((t, v))
    return events


def _basal_rate_u_per_h(t: datetime, basal: list[tuple[datetime, float]]) -> float:
    rate = basal[0][1]
    for t_change, r in basal:
        if t_change <= t:
            rate = r
    return rate


def _temp_rate_u_per_h(t: datetime, intervals: list[tuple[datetime, datetime, float]]) -> float | None:
    for a, b, rate in intervals:
        if a <= t < b:
            return rate
    return None


def _split_cgm_pairs_on_gap(
    cgm_pairs: list[tuple[datetime, float]],
    gap_minutes: float,
) -> list[list[tuple[datetime, float]]]:
    """Split sorted CGM pairs where consecutive sample times differ by more than ``gap_minutes``."""
    if not cgm_pairs:
        return []
    pairs = sorted(cgm_pairs, key=lambda x: x[0])
    segs: list[list[tuple[datetime, float]]] = [[pairs[0]]]
    for i in range(1, len(pairs)):
        dt_min = (pairs[i][0] - pairs[i - 1][0]).total_seconds() / 60.0
        if dt_min > gap_minutes:
            segs.append([pairs[i]])
        else:
            segs[-1].append(pairs[i])
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
    """One contiguous CGM segment on a 5-min grid; interpolate glucose only within this segment."""

    def minutes_from_t0(dt: datetime) -> float:
        return (dt - t0_dt).total_seconds() / 60.0

    t_end = minutes_from_t0(t_end_dt)
    n = int(np.floor((t_end - 0.0) / grid_minutes)) + 1
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
    grid_minutes: float = 5.0,
    gap_minutes: float = 60.0,
) -> list[GlucoseSeries]:
    """
    Parse OhioT1DM *-ws-{training,testing}.xml into **contiguous CGM segments**.

    Splits when consecutive CGM sample times are more than ``gap_minutes`` apart (default 1 hour).
    Each segment gets its own 5-min grid from first to last CGM in that segment; glucose is
    interpolated **only within** the segment (no bridging across long gaps or across segments).
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


def load_ohio_ws_xml(path: Path, grid_minutes: float = 5.0) -> GlucoseSeries:
    """
    Parse Ohio ws XML as a **single** series by concatenating all CGM-gap segments in file order.

    Prefer :func:`load_ohio_ws_xml_segments` for training/windowing (no cross-segment windows).
    """
    segs = load_ohio_ws_xml_segments(path, grid_minutes=grid_minutes)
    return concat_glucose_series(segs)


def load_ohio_xml(path: Path, grid_minutes: float = 5.0) -> GlucoseSeries:
    """Try Ohio ws- schema first; fall back to generic heuristic parser."""
    try:
        return load_ohio_ws_xml(path, grid_minutes=grid_minutes)
    except ValueError:
        pass
    tree = ET.parse(path)
    root = tree.getroot()
    raw = _collect_glucose_events(root)
    if len(raw) < 10:
        raise ValueError(
            f"Very few glucose events parsed from {path}. "
            "Check XML schema; you can export to CSV (timestamp,glucose,...)."
        )
    raw.sort(key=lambda x: x[0])
    t0 = raw[0][0]
    t_end = raw[-1][0]
    n = int(np.floor((t_end - t0) / grid_minutes)) + 1
    times = t0 + np.arange(n) * grid_minutes
    g = np.full(n, np.nan, dtype=np.float64)
    for t, v in raw:
        idx = int(round((t - t0) / grid_minutes))
        if 0 <= idx < n:
            g[idx] = v
    g = pd.Series(g).interpolate(limit_direction="both").to_numpy()
    ins = np.zeros_like(g)
    carb = np.zeros_like(g)
    vals = np.stack([g, ins, carb], axis=1)
    return GlucoseSeries(values=vals, index_minutes=times - t0)


def load_csv(
    path: Path,
    time_col: str = "timestamp",
    glucose_col: str = "glucose",
    insulin_col: str | None = "insulin",
    carb_col: str | None = "carbs",
    grid_minutes: float = 5.0,
) -> GlucoseSeries:
    df = pd.read_csv(path)
    if time_col not in df.columns or glucose_col not in df.columns:
        raise ValueError(f"CSV must contain {time_col} and {glucose_col}")
    df = df.sort_values(time_col)
    t = pd.to_datetime(df[time_col], utc=True, errors="coerce")
    if t.isna().all():
        mins = df[time_col].to_numpy(dtype=np.float64)
    else:
        mins = (t - t.iloc[0]).dt.total_seconds().to_numpy() / 60.0
    g = df[glucose_col].to_numpy(dtype=np.float64)
    ins = (
        df[insulin_col].to_numpy(dtype=np.float64)
        if insulin_col and insulin_col in df.columns
        else np.zeros(len(df))
    )
    carbs = (
        df[carb_col].to_numpy(dtype=np.float64)
        if carb_col and carb_col in df.columns
        else np.zeros(len(df))
    )
    df2 = pd.DataFrame({"m": mins, "g": g, "i": ins, "c": carbs}).dropna(
        subset=["m", "g"]
    )
    if df2.empty:
        raise ValueError("No rows after cleaning timestamps/glucose")
    t0 = df2["m"].iloc[0]
    t_end = df2["m"].iloc[-1]
    n = int(np.floor((t_end - t0) / grid_minutes)) + 1
    grid = t0 + np.arange(n) * grid_minutes
    out = np.zeros((n, 3), dtype=np.float64)
    mx = df2["m"].to_numpy()
    for col, j in [("g", 0), ("i", 1), ("c", 2)]:
        y = df2[col].to_numpy()
        out[:, j] = np.interp(grid, mx, y)
    return GlucoseSeries(values=out, index_minutes=grid - grid[0])


def synthesize_t1dm(
    n_steps: int = 4000,
    seed: int = 0,
    grid_minutes: float = 5.0,
) -> GlucoseSeries:
    rng = np.random.default_rng(seed)
    t = np.arange(n_steps, dtype=np.float64) * grid_minutes
    base = 120.0 + 25.0 * np.sin(2 * np.pi * t / (24 * 60))
    noise = rng.normal(0, 8.0, size=n_steps)
    meal_times = rng.choice(n_steps, size=n_steps // 400, replace=False)
    carbs = np.zeros(n_steps)
    carbs[meal_times] = rng.uniform(30, 90, size=len(meal_times))
    ins = np.maximum(0, carbs * 0.08 + rng.normal(0, 0.2, size=n_steps))
    g = base + noise
    for k in range(1, n_steps):
        g[k] += 0.85 * (g[k - 1] - g[k]) * 0.2
        g[k] += 0.35 * carbs[max(0, k - 1)]
        g[k] -= 1.2 * ins[max(0, k - 1)]
    g = np.clip(g, 50, 320)
    vals = np.stack([g, ins, carbs], axis=1)
    return GlucoseSeries(values=vals, index_minutes=t - t[0])


def concat_glucose_series(parts: list[GlucoseSeries]) -> GlucoseSeries:
    if not parts:
        raise ValueError("empty series list")
    vals = np.concatenate([p.values for p in parts], axis=0)
    mins = []
    acc = 0.0
    for p in parts:
        mins.append(p.index_minutes + acc)
        acc = float(mins[-1][-1] + 5.0)
    index = np.concatenate(mins)
    return GlucoseSeries(values=vals, index_minutes=index)


def ohio_subject_ids(training_dir: Path) -> list[str]:
    return sorted({p.name.split("-")[0] for p in Path(training_dir).glob("*-ws-training.xml")})


def load_ohio_training_segments(
    training_dir: Path,
    holdout_subject: str | None = None,
    gap_minutes: float = 60.0,
    grid_minutes: float = 5.0,
) -> list[GlucoseSeries]:
    """
    All `*-ws-training.xml` under ``training_dir`` (optional holdout exclusion), each parsed into
    **contiguous CGM segments** (split on gaps > ``gap_minutes``). Returns a flat list of segments
    from all retained subjects — **no** cross-subject concatenation for windowing.
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
        try:
            segs = load_ohio_ws_xml_segments(
                p, grid_minutes=grid_minutes, gap_minutes=gap_minutes
            )
        except ValueError:
            segs = [load_ohio_xml(p)]
        out.extend(segs)
    return out


def load_ohio_testing_subject(
    testing_dir: Path,
    subject_id: str,
    gap_minutes: float = 60.0,
    grid_minutes: float = 5.0,
) -> list[GlucoseSeries]:
    """
    Official Ohio **Testing** shard per subject: ``{{id}}-ws-testing.xml`` as **contiguous CGM segments**
    (split on gaps > ``gap_minutes``). No sliding windows should span segment boundaries.
    """
    path = Path(testing_dir) / f"{subject_id}-ws-testing.xml"
    if not path.is_file():
        raise FileNotFoundError(
            f"Expected Testing XML at {path}. Place OhioT1DM Testing export under this directory."
        )
    return load_ohio_ws_xml_segments(path, grid_minutes=grid_minutes, gap_minutes=gap_minutes)


def load_series(
    source: Literal["synthetic", "csv", "dir"] | None = None,
    path: str | Path | None = None,
) -> GlucoseSeries:
    if source is None:
        source = "synthetic"
    path = Path(path) if path else None
    if source == "synthetic":
        return synthesize_t1dm()
    if source == "csv":
        if path is None:
            raise ValueError("path required for csv")
        return load_csv(path)
    if source == "dir":
        if path is None:
            raise ValueError("path required for dir")
        xmls = sorted(path.glob("**/*.xml"))
        if not xmls:
            raise ValueError(f"No XML under {path}")
        parts: list[GlucoseSeries] = []
        for xp in xmls:
            try:
                parts.append(load_ohio_xml(xp))
            except ValueError:
                continue
        if not parts:
            raise ValueError("Could not parse glucose from any XML file")
        return concat_glucose_series(parts)
    raise ValueError(f"unknown source {source}")
