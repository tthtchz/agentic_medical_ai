
from dataclasses import dataclass

import numpy as np

from src.agent.memory import AgentMemory
from src.tools.anomaly import AnomalyResult


@dataclass
class PlannedAction:
    use_lstm: bool
    use_mc_dropout: bool
    use_guideline: bool
    rationale: str


def plan_step(
    window: np.ndarray,
    anomaly: AnomalyResult,
    memory: AgentMemory,
) -> PlannedAction:
    g = window[:, 0]
    baseline = float(g[-1])
    roc = float(np.mean(np.diff(g[-4:]))) if len(g) >= 4 else 0.0
    mae = memory.recent_mae()

    if baseline < 80 or baseline > 240:
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

    volatile = abs(roc) > 1.4
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

    if memory.steps_since_deep > 18:
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
