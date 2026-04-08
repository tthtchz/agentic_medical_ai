"""Rolling forecast-error buffer and adaptive MAE trigger for ``policy.plan_step``."""

from collections import deque
from dataclasses import dataclass, field

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------

_DEFAULT_ERROR_WINDOW = 36
_DEFAULT_MAE_TRIGGER_MGDL = 22.0
_DEFAULT_MAE_FLOOR_MGDL = 12.0
_DEFAULT_DEEP_REFRESH_INTERVAL = 18

# ``reflect()``: recent MAE above this → lower trigger (prefer LSTM more often).
_REFLECT_MAE_HIGH_MGDL = 28.0
# Same threshold as original ``m < 12.0`` relax branch (independent of custom ``mae_floor_mgdl``).
_REFLECT_RELAX_BELOW_MGDL = 12.0
_REFLECT_TRIGGER_DECREMENT_MGDL = 3.0
_REFLECT_TRIGGER_INCREMENT_MGDL = 1.0

# -----------------------------------------------------------------------------
# Agent memory
# -----------------------------------------------------------------------------


@dataclass
class AgentMemory:
    """Rolling absolute-error window (mg/dL), adaptive MAE trigger, and deep-call cadence."""

    error_window: int = _DEFAULT_ERROR_WINDOW
    mae_trigger_mgdl: float = _DEFAULT_MAE_TRIGGER_MGDL
    mae_floor_mgdl: float = _DEFAULT_MAE_FLOOR_MGDL
    deep_refresh_interval_steps: int = _DEFAULT_DEEP_REFRESH_INTERVAL
    steps_since_deep: int = 0
    abs_errors_mgdl: deque[float] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.error_window < 1:
            raise ValueError("error_window must be >= 1")
        if self.deep_refresh_interval_steps < 1:
            raise ValueError("deep_refresh_interval_steps must be >= 1")
        if self.mae_trigger_mgdl < self.mae_floor_mgdl:
            raise ValueError("mae_trigger_mgdl must be >= mae_floor_mgdl")
        object.__setattr__(
            self,
            "abs_errors_mgdl",
            deque(maxlen=self.error_window),
        )

    def push_error(self, abs_err: float) -> None:
        self.abs_errors_mgdl.append(float(abs_err))

    def mark_deep_call(self) -> None:
        self.steps_since_deep = 0

    def tick_cheap_step(self) -> None:
        self.steps_since_deep += 1

    def recent_mae(self) -> float | None:
        errs = self.abs_errors_mgdl
        if not errs:
            return None
        n = len(errs)
        return float(sum(errs) / n)

    def reflect(self) -> None:
        """Lower ``mae_trigger_mgdl`` if recent MAE is high; raise it slightly if MAE is very low."""
        m = self.recent_mae()
        if m is None:
            return
        if m > _REFLECT_MAE_HIGH_MGDL:
            self.mae_trigger_mgdl = max(
                self.mae_floor_mgdl,
                self.mae_trigger_mgdl - _REFLECT_TRIGGER_DECREMENT_MGDL,
            )
        elif (
            m < _REFLECT_RELAX_BELOW_MGDL
            and self.mae_trigger_mgdl < _DEFAULT_MAE_TRIGGER_MGDL
        ):
            self.mae_trigger_mgdl = min(
                _DEFAULT_MAE_TRIGGER_MGDL,
                self.mae_trigger_mgdl + _REFLECT_TRIGGER_INCREMENT_MGDL,
            )
