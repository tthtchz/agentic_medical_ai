
from collections import deque
from dataclasses import dataclass, field


@dataclass
class AgentMemory:
    """Short-term numeric memory + reflection knobs."""

    error_window: int = 36
    abs_errors_mgdl: deque[float] = field(default_factory=lambda: deque(maxlen=36))
    mae_trigger_mgdl: float = 22.0
    mae_floor_mgdl: float = 12.0
    steps_since_deep: int = 0

    def push_error(self, abs_err: float) -> None:
        self.abs_errors_mgdl.append(abs_err)

    def mark_deep_call(self) -> None:
        self.steps_since_deep = 0

    def tick_cheap_step(self) -> None:
        self.steps_since_deep += 1

    def recent_mae(self) -> float | None:
        if not self.abs_errors_mgdl:
            return None
        return float(sum(self.abs_errors_mgdl) / len(self.abs_errors_mgdl))

    def reflect(self) -> None:
        """Tighten deep-model usage if we have been systematically wrong."""
        m = self.recent_mae()
        if m is None:
            return
        if m > 28.0:
            self.mae_trigger_mgdl = max(self.mae_floor_mgdl, self.mae_trigger_mgdl - 3.0)
        elif m < 12.0 and self.mae_trigger_mgdl < 22.0:
            self.mae_trigger_mgdl = min(22.0, self.mae_trigger_mgdl + 1.0)
