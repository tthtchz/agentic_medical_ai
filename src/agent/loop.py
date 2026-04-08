"""Agent rollouts: fit anomaly tool on training windows, run policy + forecaster on held-out time."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.agent.memory import AgentMemory
from src.agent.policy import plan_step
from src.data.dataset import GlucoseSeries
from src.tools.anomaly import MultivariateAnomalyTool
from src.tools.forecast import LstmForecastTool
from src.tools.guidelines import GuidelineRetrievalTool

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------

# Minimum train windows for ``MultivariateAnomalyTool.fit`` in train/test split mode.
_MIN_TRAIN_WINDOWS = 10
# Ensure enough past + future steps inside the train slice (single-series split).
_EXTRA_STEPS_BEYOND_LOOKBACK_HORIZON = 50
# Truncate guideline text for trajectory logging.
_GUIDELINE_SNIPPET_MAX_LEN = 180
# Subsample training windows for IsolationForest when above ``cfg.anomaly_fit_max_windows``.
_ANOMALY_FIT_SUBSAMPLE_SEED = 0


# -----------------------------------------------------------------------------
# Config and trajectory record
# -----------------------------------------------------------------------------


@dataclass
class AgentConfig:
    """Checkpoint path, train fraction, and anomaly-tool limits."""

    ckpt_path: Path
    train_fraction: float = 0.7
    anomaly_contamination: float = 0.06
    # Cap windows passed to ``IsolationForest.fit`` (full Ohio train is 100k+ slices).
    anomaly_fit_max_windows: int = 8000


@dataclass
class AgentTrajectoryStep:
    """One step along the evaluated horizon: prediction, tools used, policy rationale."""

    t_index: int
    predicted_glucose: float
    actual_glucose: float
    used_lstm: bool
    used_mc: bool
    used_guideline: bool
    anomaly_ood: bool
    rationale: str
    guideline_snippet: str | None


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def _segment_list(x: GlucoseSeries | list[GlucoseSeries]) -> list[GlucoseSeries]:
    return x if isinstance(x, list) else [x]


def _maybe_subsample_windows(windows: np.ndarray, max_windows: int) -> np.ndarray:
    n = windows.shape[0]
    if n <= max_windows:
        return windows
    rng = np.random.default_rng(_ANOMALY_FIT_SUBSAMPLE_SEED)
    sel = rng.choice(n, size=max_windows, replace=False)
    return windows[sel]


def _assert_forecaster_matches_ckpt(
    forecaster: LstmForecastTool,
    lookback: int,
    horizon: int,
) -> None:
    if forecaster.lookback != lookback:
        raise ValueError(
            f"Checkpoint lookback {forecaster.lookback} != {lookback}; "
            "retrain with matching lookback."
        )
    if forecaster.horizon != horizon:
        raise ValueError(
            f"Checkpoint horizon {forecaster.horizon} != {horizon}; "
            "retrain with matching horizon."
        )


def _one_step(
    t_index: int,
    window: np.ndarray,
    true_future: float,
    anomaly_tool: MultivariateAnomalyTool,
    forecaster: LstmForecastTool,
    guideline_tool: GuidelineRetrievalTool,
    memory: AgentMemory,
) -> AgentTrajectoryStep:
    ar = anomaly_tool.score(window)
    action = plan_step(window, ar, memory)

    g_line: str | None = None
    pred = float(window[-1, 0])
    used_mc = False

    if action.use_lstm:
        fr = forecaster.predict_window(window, mc=action.use_mc_dropout)
        pred = fr.glucose_mgdl
        used_mc = fr.used_dropout_mc
        memory.mark_deep_call()
    else:
        memory.tick_cheap_step()
    if action.use_guideline:
        hit = guideline_tool.query(pred)
        g_line = hit.text[:_GUIDELINE_SNIPPET_MAX_LEN]

    memory.push_error(abs(pred - true_future))
    memory.reflect()

    return AgentTrajectoryStep(
        t_index=t_index,
        predicted_glucose=pred,
        actual_glucose=true_future,
        used_lstm=action.use_lstm,
        used_mc=used_mc,
        used_guideline=action.use_guideline,
        anomaly_ood=ar.is_ood,
        rationale=action.rationale,
        guideline_snippet=g_line,
    )


# -----------------------------------------------------------------------------
# Runs
# -----------------------------------------------------------------------------


def run_agent_on_series(
    series: GlucoseSeries,
    cfg: AgentConfig,
    lookback: int,
    horizon: int,
) -> tuple[list[AgentTrajectoryStep], AgentMemory]:
    """
    Time-split one series: train portion builds anomaly windows; test portion runs the agent.

    If the train slice would be shorter than ``lookback + horizon + 50`` (see
    ``_EXTRA_STEPS_BEYOND_LOOKBACK_HORIZON``), the split is expanded to that minimum.
    """
    vals = series.values.astype(np.float64)
    t_end, _ = vals.shape
    split = int(cfg.train_fraction * t_end)
    min_split = lookback + horizon + _EXTRA_STEPS_BEYOND_LOOKBACK_HORIZON
    if split < min_split:
        split = min_split
    train_vals = vals[:split]
    test_start = split

    windows_train: list[np.ndarray] = []
    for t in range(lookback, split - horizon):
        windows_train.append(train_vals[t - lookback : t])
    windows_train_arr = np.stack(windows_train)
    windows_train_arr = _maybe_subsample_windows(
        windows_train_arr, cfg.anomaly_fit_max_windows
    )

    anomaly_tool = MultivariateAnomalyTool(contamination=cfg.anomaly_contamination)
    anomaly_tool.fit(windows_train_arr)

    forecaster = LstmForecastTool(cfg.ckpt_path)
    _assert_forecaster_matches_ckpt(forecaster, lookback, horizon)

    guideline_tool = GuidelineRetrievalTool()
    memory = AgentMemory()
    traj: list[AgentTrajectoryStep] = []

    for t in range(test_start, t_end - horizon):
        window = vals[t - lookback : t]
        true_future = float(vals[t + horizon, 0])
        traj.append(
            _one_step(
                t,
                window,
                true_future,
                anomaly_tool,
                forecaster,
                guideline_tool,
                memory,
            )
        )

    return traj, memory


def run_agent_on_train_test(
    train: GlucoseSeries | list[GlucoseSeries],
    test: GlucoseSeries | list[GlucoseSeries],
    cfg: AgentConfig,
    lookback: int,
    horizon: int,
    max_test_steps: int | None = None,
) -> tuple[list[AgentTrajectoryStep], AgentMemory]:
    """
    Fit the anomaly tool on ``train`` windows only; run the agent on ``test`` (official Ohio-style split).

    Pass lists of segments so windows do not span CGM gaps or subject boundaries.
    """
    train_segs = _segment_list(train)
    test_segs = _segment_list(test)

    windows_train: list[np.ndarray] = []
    for tr_series in train_segs:
        tr = tr_series.values.astype(np.float64)
        for t in range(lookback, tr.shape[0] - horizon):
            windows_train.append(tr[t - lookback : t])
    if len(windows_train) < _MIN_TRAIN_WINDOWS:
        raise ValueError("Training series too short for anomaly tool.")
    windows_train_arr = np.stack(windows_train)
    windows_train_arr = _maybe_subsample_windows(
        windows_train_arr, cfg.anomaly_fit_max_windows
    )

    anomaly_tool = MultivariateAnomalyTool(contamination=cfg.anomaly_contamination)
    anomaly_tool.fit(windows_train_arr)

    forecaster = LstmForecastTool(cfg.ckpt_path)
    _assert_forecaster_matches_ckpt(forecaster, lookback, horizon)

    guideline_tool = GuidelineRetrievalTool()
    memory = AgentMemory()
    traj: list[AgentTrajectoryStep] = []

    remaining = max_test_steps
    time_base = 0
    for te_series in test_segs:
        te = te_series.values.astype(np.float64)
        tt = te.shape[0]
        for t in range(lookback, tt - horizon):
            if remaining is not None and remaining <= 0:
                break
            window = te[t - lookback : t]
            true_future = float(te[t + horizon, 0])
            traj.append(
                _one_step(
                    time_base + t,
                    window,
                    true_future,
                    anomaly_tool,
                    forecaster,
                    guideline_tool,
                    memory,
                )
            )
            if remaining is not None:
                remaining -= 1
        if remaining is not None and remaining <= 0:
            break
        time_base += tt

    return traj, memory
