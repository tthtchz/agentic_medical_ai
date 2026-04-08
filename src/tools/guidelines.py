
from dataclasses import dataclass


@dataclass
class GuidelineHit:
    text: str
    urgency: str


class GuidelineRetrievalTool:
    """Lightweight rule-based retrieval (stand-in for RAG) using CGM-focused thresholds."""

    def __init__(self) -> None:
        self.entries = [
            (
                lambda g: g < 70,
                "HYPO",
                "Predicted or observed glucose below 70 mg/dL: prioritize hypoglycemia mitigation per clinical protocol; confirm with fingerstick if available.",
            ),
            (
                lambda g: 70 <= g < 100,
                "LOW_NORMAL",
                "Borderline low: consider reducing insulin sensitivity and monitor closely over next 30–60 minutes.",
            ),
            (
                lambda g: g > 250,
                "HIGH_SEVERE",
                "Very high glucose: consider ketone check and escalation per care plan; verify infusion site and recent carbs/insulin.",
            ),
            (
                lambda g: 180 < g <= 250,
                "HIGH",
                "Above typical ambulatory target: review recent meals/bolus timing; hydration and correction per protocol.",
            ),
            (
                lambda g: 100 <= g <= 180,
                "IN_RANGE",
                "Within common ambulatory CGM target band for many adults; continue routine monitoring.",
            ),
        ]

    def query(self, glucose_mgdl: float) -> GuidelineHit:
        for fn, label, text in self.entries:
            if fn(glucose_mgdl):
                urg = "high" if label in ("HYPO", "HIGH_SEVERE") else "medium" if "HIGH" in label else "low"
                return GuidelineHit(text=text, urgency=urg)
        return GuidelineHit(text="No matching guideline.", urgency="low")
