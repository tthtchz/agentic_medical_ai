"""IsolationForest on hand-crafted window stats — OOD / distribution-shift proxy."""

from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# -----------------------------------------------------------------------------
# Defaults (sklearn)
# -----------------------------------------------------------------------------

_N_ESTIMATORS = 100
_SKLEARN_OUTLIER_LABEL = -1

# -----------------------------------------------------------------------------
# Result type
# -----------------------------------------------------------------------------


@dataclass
class AnomalyResult:
    """IsolationForest decision score; ``is_ood`` when sklearn predicts outlier."""

    score: float
    is_ood: bool
    detail: str


# -----------------------------------------------------------------------------
# Tool
# -----------------------------------------------------------------------------


class MultivariateAnomalyTool:
    """
    Fit on training lag windows; score new windows by how far their **scalar features**
    depart from the training bulk (IsolationForest + StandardScaler).

    Features per window: glucose mean/std, diff mean/std, last row (multivariate snapshot).
    """

    def __init__(self, contamination: float = 0.05, random_state: int = 0) -> None:
        self._scaler = StandardScaler()
        self._forest = IsolationForest(
            n_estimators=_N_ESTIMATORS,
            contamination=contamination,
            random_state=random_state,
        )
        self._fitted = False

    @staticmethod
    def featurize_window(window: np.ndarray) -> np.ndarray:
        """Map ``(lookback, F)`` window to one row of shape ``(1, n_feats)``."""
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
        """``windows`` iterable of 2D arrays ``(lookback, F)`` (same as agent training loop)."""
        X = np.vstack([self.featurize_window(w) for w in windows])
        Xs = self._scaler.fit_transform(X)
        self._forest.fit(Xs)
        self._fitted = True

    def score(self, window: np.ndarray) -> AnomalyResult:
        if not self._fitted:
            return AnomalyResult(score=0.0, is_ood=False, detail="not_fitted")

        x = self.featurize_window(window)
        xs = self._scaler.transform(x)
        s = float(self._forest.decision_function(xs)[0])
        label = int(self._forest.predict(xs)[0])
        is_ood = label == _SKLEARN_OUTLIER_LABEL
        return AnomalyResult(
            score=s,
            is_ood=is_ood,
            detail="isolation_forest",
        )
