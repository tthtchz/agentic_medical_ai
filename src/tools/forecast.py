"""LSTM checkpoint loader: one-step glucose forecast (mg/dL) at the trained horizon."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from src.models.lstm_predictor import GlucoseLSTM

# -----------------------------------------------------------------------------
# Result type
# -----------------------------------------------------------------------------


@dataclass
class ForecastResult:
    """Glucose at ``t + horizon`` (training horizon); optional MC-dropout uncertainty."""

    glucose_mgdl: float
    uncertainty_mgdl: float
    used_dropout_mc: bool


# -----------------------------------------------------------------------------
# Tool
# -----------------------------------------------------------------------------


class LstmForecastTool:
    """
    Load ``GlucoseLSTM`` from ``train_lstm.py`` checkpoint (``state``, ``norm_*``, shapes).

    **Input:** one window ``(lookback, n_features)`` in **original units** (Ohio: glucose, insulin,
    carbs). Same order as training.

    **Output:** scalar glucose at horizon (e.g. 6 steps @ 5 min = 30 min ahead). Optional MC
    dropout forward passes approximate epistemic spread (denormalized to mg/dL on the glucose channel).
    """

    def __init__(
        self,
        ckpt_path: Path | str,
        device: torch.device | None = None,
        mc_samples: int = 4,
    ) -> None:
        ckpt_path = Path(ckpt_path)
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

        self.lookback = int(ckpt["lookback"])
        self.horizon = int(ckpt["horizon_steps"])
        self.n_features = int(ckpt["n_features"])

        mean = ckpt.get("norm_mean")
        std = ckpt.get("norm_std")
        if mean is not None and std is not None:
            self.mean = np.asarray(mean, dtype=np.float64)
            self.std = np.asarray(std, dtype=np.float64)
        else:
            self.mean = None
            self.std = None

        self.model = GlucoseLSTM(
            n_features=self.n_features,
            horizon_steps=self.horizon,
        )
        self.model.load_state_dict(ckpt["state"])
        self.model.eval()
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model.to(self.device)
        self.mc_samples = max(1, int(mc_samples))

    def _normalize(self, window: np.ndarray) -> np.ndarray:
        if self.mean is None or self.std is None:
            return window.astype(np.float64)
        return (window.astype(np.float64) - self.mean) / self.std

    def _denorm_glucose(self, g_norm: float, unc_norm: float) -> tuple[float, float]:
        if self.mean is None or self.std is None:
            return float(g_norm), float(unc_norm)
        scale = float(self.std[0, 0])
        offset = float(self.mean[0, 0])
        return g_norm * scale + offset, unc_norm * scale

    def _check_window(self, window: np.ndarray) -> None:
        if window.ndim != 2:
            raise ValueError(
                f"window must be 2D (lookback, n_features), got shape {window.shape}"
            )
        if window.shape[0] != self.lookback:
            raise ValueError(
                f"window length {window.shape[0]} != checkpoint lookback {self.lookback}"
            )
        if window.shape[1] != self.n_features:
            raise ValueError(
                f"window has {window.shape[1]} features, checkpoint expects {self.n_features}"
            )

    def predict_window(self, window: np.ndarray, mc: bool) -> ForecastResult:
        """
        Parameters
        ----------
        window
            ``(lookback, n_features)`` in original units.
        mc
            If True and ``mc_samples`` > 1, run stochastic dropout forwards and return uncertainty.
        """
        self._check_window(window)
        x = self._normalize(window[None, ...])
        xt = torch.from_numpy(x).float().to(self.device)
        use_mc = bool(mc) and self.mc_samples > 1

        if use_mc:
            mean_t, std_t = self.model.predict_with_uncertainty(
                xt,
                n_samples=self.mc_samples,
                dropout_at_inference=True,
            )
            g_n = float(mean_t.reshape(-1)[0].item())
            u_n = float(std_t.reshape(-1)[0].item())
        else:
            with torch.no_grad():
                pred = self.model(xt)
            g_n = float(pred.reshape(-1)[0].item())
            u_n = 0.0

        g, u = self._denorm_glucose(g_n, u_n)
        return ForecastResult(
            glucose_mgdl=g,
            uncertainty_mgdl=u,
            used_dropout_mc=use_mc,
        )
