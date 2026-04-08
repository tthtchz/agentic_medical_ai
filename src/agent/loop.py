
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from src.agent.memory import AgentMemory
from src.agent.policy import plan_step
from src.data.dataset import GlucoseSeries
from src.tools.anomaly import MultivariateAnomalyTool
from src.tools.forecast import LstmForecastTool
from src.tools.guidelines import GuidelineRetrievalTool


@dataclass
class AgentConfig:
    ckpt_path: Path
    train_fraction: float = 0.7
    anomaly_contamination: float = 0.06
    """Cap windows passed to IsolationForest.fit (full Ohio train is 100k+ slices)."""
    anomaly_fit_max_windows: int = 8000


@dataclass
class AgentTrajectoryStep:
    t_index: int
    predicted_glucose: float
    actual_glucose: float
    used_lstm: bool
    used_mc: bool
    used_guideline: bool
    anomaly_ood: bool
    rationale: str
    guideline_snippet: str | None


def run_agent_on_series(
    series: GlucoseSeries,
    cfg: AgentConfig,
    lookback: int,
    horizon: int,
) -> tuple[list[AgentTrajectoryStep], AgentMemory]:
    vals = series.values.astype(np.float64)
    T, Fdim = vals.shape
    split = int(cfg.train_fraction * T)
    if split < lookback + horizon + 50:
        split = lookback + horizon + 50
    train_vals = vals[:split]
    test_start = split

    windows_train = []
    for t in range(lookback, split - horizon):
        windows_train.append(train_vals[t - lookback : t])
    windows_train = np.stack(windows_train)
    if windows_train.shape[0] > cfg.anomaly_fit_max_windows:
        rng = np.random.default_rng(0)
        sel = rng.choice(
            windows_train.shape[0],
            size=cfg.anomaly_fit_max_windows,
            replace=False,
        )
        windows_train = windows_train[sel]

    anomaly_tool = MultivariateAnomalyTool(contamination=cfg.anomaly_contamination)
    anomaly_tool.fit(windows_train)

    forecaster = LstmForecastTool(cfg.ckpt_path)
    if forecaster.lookback != lookback:
        raise ValueError(
            f"Checkpoint lookback {forecaster.lookback} != {lookback}; retrain with matching lookback."
        )
    if forecaster.horizon != horizon:
        raise ValueError(
            f"Checkpoint horizon {forecaster.horizon} != {horizon}; retrain with matching horizon."
        )

    guideline_tool = GuidelineRetrievalTool()
    memory = AgentMemory()
    traj: list[AgentTrajectoryStep] = []

    for t in range(test_start, T - horizon):
        window = vals[t - lookback : t]
        true_future = float(vals[t + horizon, 0])
        ar = anomaly_tool.score(window)
        action = plan_step(window, ar, memory)

        g_line = None
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
            g_line = hit.text[:180]

        abs_err = abs(pred - true_future)
        memory.push_error(abs_err)
        memory.reflect()

        traj.append(
            AgentTrajectoryStep(
                t_index=t,
                predicted_glucose=pred,
                actual_glucose=true_future,
                used_lstm=action.use_lstm,
                used_mc=used_mc,
                used_guideline=action.use_guideline,
                anomaly_ood=ar.is_ood,
                rationale=action.rationale,
                guideline_snippet=g_line,
            )
        )

    return traj, memory


def _segment_list(x: GlucoseSeries | list[GlucoseSeries]) -> list[GlucoseSeries]:
    return x if isinstance(x, list) else [x]


def run_agent_on_train_test(
    train: GlucoseSeries | list[GlucoseSeries],
    test: GlucoseSeries | list[GlucoseSeries],
    cfg: AgentConfig,
    lookback: int,
    horizon: int,
    max_test_steps: int | None = None,
) -> tuple[list[AgentTrajectoryStep], AgentMemory]:
    """Fit anomaly detector on `train`; run the agent only on `test` (official Ohio split).

    Pass a list of contiguous segments so train/test windows never span CGM gaps or subject boundaries.
    """
    train_segs = _segment_list(train)
    test_segs = _segment_list(test)

    windows_train: list[np.ndarray] = []
    for tr_series in train_segs:
        tr = tr_series.values.astype(np.float64)
        for t in range(lookback, tr.shape[0] - horizon):
            windows_train.append(tr[t - lookback : t])
    if len(windows_train) < 10:
        raise ValueError("Training series too short for anomaly tool.")
    windows_train_arr = np.stack(windows_train)
    if windows_train_arr.shape[0] > cfg.anomaly_fit_max_windows:
        rng = np.random.default_rng(0)
        sel = rng.choice(
            windows_train_arr.shape[0],
            size=cfg.anomaly_fit_max_windows,
            replace=False,
        )
        windows_train_arr = windows_train_arr[sel]

    anomaly_tool = MultivariateAnomalyTool(contamination=cfg.anomaly_contamination)
    anomaly_tool.fit(windows_train_arr)

    forecaster = LstmForecastTool(cfg.ckpt_path)
    if forecaster.lookback != lookback:
        raise ValueError(
            f"Checkpoint lookback {forecaster.lookback} != {lookback}; retrain with matching lookback."
        )
    if forecaster.horizon != horizon:
        raise ValueError(
            f"Checkpoint horizon {forecaster.horizon} != {horizon}; retrain with matching horizon."
        )

    guideline_tool = GuidelineRetrievalTool()
    memory = AgentMemory()
    traj: list[AgentTrajectoryStep] = []

    remaining = max_test_steps
    time_base = 0
    for te_series in test_segs:
        te = te_series.values.astype(np.float64)
        Tt = te.shape[0]
        for t in range(lookback, Tt - horizon):
            if max_test_steps is not None and remaining is not None and remaining <= 0:
                break
            window = te[t - lookback : t]
            true_future = float(te[t + horizon, 0])
            ar = anomaly_tool.score(window)
            action = plan_step(window, ar, memory)

            g_line = None
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
                g_line = hit.text[:180]

            memory.push_error(abs(pred - true_future))
            memory.reflect()

            traj.append(
                AgentTrajectoryStep(
                    t_index=time_base + t,
                    predicted_glucose=pred,
                    actual_glucose=true_future,
                    used_lstm=action.use_lstm,
                    used_mc=used_mc,
                    used_guideline=action.use_guideline,
                    anomaly_ood=ar.is_ood,
                    rationale=action.rationale,
                    guideline_snippet=g_line,
                )
            )
            if max_test_steps is not None and remaining is not None:
                remaining -= 1
        if max_test_steps is not None and remaining is not None and remaining <= 0:
            break
        time_base += Tt

    return traj, memory
