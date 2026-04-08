
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from src.models.lstm_predictor import GlucoseLSTM


@dataclass
class ForecastResult:
    """Single-step glucose forecast at the trained horizon (same as `horizon_steps` in checkpoint)."""

    glucose_mgdl: float
    uncertainty_mgdl: float
    used_dropout_mc: bool


class LstmForecastTool:
    """
    Loads `GlucoseLSTM` from `scripts/train_lstm.py` checkpoint and predicts future glucose (mg/dL).

    - **Input window** shape must be `(lookback, n_features)` with the same channels/order as training
      (Ohio: glucose, insulin, carbs).
    - **Output** is one scalar: model is trained to predict glucose at `t + horizon` (e.g. 6 steps = 30 min).
    - Optional **MC dropout** passes for epistemic uncertainty (std dev in normalized space, scaled to mg/dL).
    """

    def __init__(
        self,
        ckpt_path: Path,
        device: torch.device | None = None,
        mc_samples: int = 4,
    ):
        ckpt_path = Path(ckpt_path)
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        n_features = int(ckpt["n_features"])
        self.lookback = int(ckpt["lookback"])
        self.horizon = int(ckpt["horizon_steps"])
        self.n_features = n_features
        self.mean = ckpt.get("norm_mean")
        self.std = ckpt.get("norm_std")
        if self.mean is not None:
            self.mean = np.asarray(self.mean, dtype=np.float64)
            self.std = np.asarray(self.std, dtype=np.float64)
        self.model = GlucoseLSTM(
            n_features=n_features,
            horizon_steps=self.horizon,
        )
        self.model.load_state_dict(ckpt["state"])
        self.model.eval()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.mc_samples = max(1, int(mc_samples))

    def _normalize(self, window: np.ndarray) -> np.ndarray:
        if self.mean is None:
            return window.astype(np.float64)
        return (window - self.mean) / self.std

    def _denorm_glucose(self, g_norm: float, unc_norm: float) -> tuple[float, float]:
        if self.mean is None:
            return float(g_norm), float(unc_norm)
        s = float(self.std[0, 0])
        m = float(self.mean[0, 0])
        return g_norm * s + m, unc_norm * s

    def _check_window(self, window: np.ndarray) -> None:
        if window.ndim != 2:
            raise ValueError(f"window must be 2D (lookback, n_features), got shape {window.shape}")
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
        Predict glucose at horizon from one lag window.

        Parameters
        ----------
        window : (lookback, n_features)
        mc : bool
            If True and ``mc_samples`` > 1, enable dropout at inference and return MC spread as uncertainty.
        """
        self._check_window(window)
        x = self._normalize(window[None, ...])
        xt = torch.from_numpy(x).float().to(self.device)
        use_mc = bool(mc) and self.mc_samples > 1
        if use_mc:
            mean, std = self.model.predict_with_uncertainty(
                xt, n_samples=self.mc_samples, dropout_at_inference=True
            )
            g_n = float(mean.reshape(-1)[0].item())
            u_n = float(std.reshape(-1)[0].item())
        else:
            with torch.no_grad():
                pred = self.model(xt)
            g_n = float(pred.reshape(-1)[0].item())
            u_n = 0.0
        g, u = self._denorm_glucose(g_n, u_n)
        return ForecastResult(glucose_mgdl=g, uncertainty_mgdl=u, used_dropout_mc=use_mc)
