"""Rule-based glucose band → short text (placeholder for real guideline / RAG retrieval)."""

from dataclasses import dataclass
from typing import Callable


@dataclass
class GuidelineHit:
    """One snippet plus coarse urgency for logging or UI."""

    text: str
    urgency: str


def _urgency_for_label(label: str) -> str:
    if label in ("HYPO", "HIGH_SEVERE"):
        return "high"
    if "HIGH" in label:
        return "medium"
    return "low"


# (predicate, stable_label, user-facing text) — first match wins; order matters (hypo / severe first).
_GUIDELINE_RULES: list[
    tuple[Callable[[float], bool], str, str]
] = [
    (
        lambda g: g < 70,
        "HYPO",
        "Predicted or observed glucose below 70 mg/dL: prioritize hypoglycemia mitigation per "
        "clinical protocol; confirm with fingerstick if available.",
    ),
    (
        lambda g: 70 <= g < 100,
        "LOW_NORMAL",
        "Borderline low: consider reducing insulin sensitivity and monitor closely over next "
        "30–60 minutes.",
    ),
    (
        lambda g: g > 250,
        "HIGH_SEVERE",
        "Very high glucose: consider ketone check and escalation per care plan; verify infusion "
        "site and recent carbs/insulin.",
    ),
    (
        lambda g: 180 < g <= 250,
        "HIGH",
        "Above typical ambulatory target: review recent meals/bolus timing; hydration and correction "
        "per protocol.",
    ),
    (
        lambda g: 100 <= g <= 180,
        "IN_RANGE",
        "Within common ambulatory CGM target band for many adults; continue routine monitoring.",
    ),
]


class GuidelineRetrievalTool:
    """
    Map a single glucose value (mg/dL) to a fixed string by threshold bands.

    Not a retrieval system over documents — a **stand-in** for RAG / clinical KB integration.
    """

    def __init__(self) -> None:
        self._rules = _GUIDELINE_RULES

    def query(self, glucose_mgdl: float) -> GuidelineHit:
        for pred, label, text in self._rules:
            if pred(glucose_mgdl):
                return GuidelineHit(
                    text=text,
                    urgency=_urgency_for_label(label),
                )
        return GuidelineHit(text="No matching guideline.", urgency="low")
