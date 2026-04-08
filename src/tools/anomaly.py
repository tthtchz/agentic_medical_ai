
from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


@dataclass
class AnomalyResult:
    score: float
    is_ood: bool
    detail: str


class MultivariateAnomalyTool:
    """Flags windows whose feature distribution departs from training (proxy for OOD)."""

    def __init__(self, contamination: float = 0.05, random_state: int = 0):
        self.scaler = StandardScaler()
        self.model = IsolationForest(
            n_estimators=100,
            contamination=contamination,
            random_state=random_state,
        )
        self._fitted = False

    @staticmethod
    def featurize_window(window: np.ndarray) -> np.ndarray:
        g = window[:, 0]
        dg = np.diff(g, prepend=g[0])
        feats = np.concatenate(
            [
                g.mean(keepdims=True),
                g.std(keepdims=True),
                dg.mean(keepdims=True),
                dg.std(keepdims=True),
                window[-1],
            ]
        )
        return feats.reshape(1, -1)

    def fit(self, windows: np.ndarray) -> None:
        X = np.vstack([self.featurize_window(w) for w in windows])
        Xs = self.scaler.fit_transform(X)
        self.model.fit(Xs)
        self._fitted = True

    def score(self, window: np.ndarray) -> AnomalyResult:
        if not self._fitted:
            return AnomalyResult(score=0.0, is_ood=False, detail="not_fitted")
        x = self.featurize_window(window)
        xs = self.scaler.transform(x)
        s = float(self.model.decision_function(xs)[0])
        pred = int(self.model.predict(xs)[0])
        is_ood = pred == -1
        detail = "isolation_forest"
        return AnomalyResult(score=s, is_ood=is_ood, detail=detail)
