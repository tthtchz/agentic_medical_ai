"""Heuristic step policy: when to call LSTM, MC dropout, and guideline tools given window + memory."""

from dataclasses import dataclass

import numpy as np

from src.agent.memory import AgentMemory
from src.tools.anomaly import AnomalyResult

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------

# Last CGM in hypo / severe hyper → always escalate (mg/dL).
_CRITICAL_LOW_MGDL = 80.0
_CRITICAL_HIGH_MGDL = 240.0

# Short tail for rate-of-change; ``diff`` yields ``n-1`` steps.
_ROC_TAIL_POINTS = 4
_VOLATILE_ABS_ROC_THRESHOLD = 1.4


# -----------------------------------------------------------------------------
# Planned action
# -----------------------------------------------------------------------------


@dataclass
class PlannedAction:
    """One agent step: which tools to run and a short human-readable rationale."""

    use_lstm: bool
    use_mc_dropout: bool
    use_guideline: bool
    rationale: str


# -----------------------------------------------------------------------------
# Policy
# -----------------------------------------------------------------------------


def plan_step(
    window: np.ndarray,
    anomaly: AnomalyResult,
    memory: AgentMemory,
) -> PlannedAction:
    """
    Decide tools for this step. Rules are evaluated in order; first match wins.

    Uses :class:`AgentMemory` for ``recent_mae`` vs ``mae_trigger_mgdl`` and
    ``steps_since_deep`` vs ``deep_refresh_interval_steps``.
    """
    g = window[:, 0]
    baseline = float(g[-1])
    if len(g) >= _ROC_TAIL_POINTS:
        roc = float(np.mean(np.diff(g[-_ROC_TAIL_POINTS:])))
    else:
        roc = 0.0
    mae = memory.recent_mae()

    if baseline < _CRITICAL_LOW_MGDL or baseline > _CRITICAL_HIGH_MGDL:
        return PlannedAction(
            use_lstm=True,
            use_mc_dropout=True,
            use_guideline=True,
            rationale="Critical range on last CGM; deep forecast + guideline retrieval.",
        )

    if anomaly.is_ood:
        return PlannedAction(
            use_lstm=True,
            use_mc_dropout=True,
            use_guideline=True,
            rationale="OOD window from isolation forest; escalate forecasting + retrieval.",
        )

    volatile = abs(roc) > _VOLATILE_ABS_ROC_THRESHOLD
    if volatile:
        return PlannedAction(
            use_lstm=True,
            use_mc_dropout=True,
            use_guideline=False,
            rationale="High short-term rate of change; model + uncertainty.",
        )

    if mae is not None and mae > memory.mae_trigger_mgdl:
        return PlannedAction(
            use_lstm=True,
            use_mc_dropout=True,
            use_guideline=False,
            rationale="Reflection: recent MAE above adaptive trigger; call deep forecaster.",
        )

    if memory.steps_since_deep > memory.deep_refresh_interval_steps:
        return PlannedAction(
            use_lstm=True,
            use_mc_dropout=False,
            use_guideline=False,
            rationale="Periodic deep refresh after many cheap steps.",
        )

    return PlannedAction(
        use_lstm=False,
        use_mc_dropout=False,
        use_guideline=False,
        rationale="Stable regime: persistence baseline suffices.",
    )
