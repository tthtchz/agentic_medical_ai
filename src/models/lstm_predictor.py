
import math
from pathlib import Path

import torch
import torch.nn as nn


class GlucoseLSTM(nn.Module):
    """Multivariate LSTM mapping a lag window to a scalar glucose horizon prediction."""

    def __init__(
        self,
        n_features: int,
        hidden: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
        horizon_steps: int = 6,
    ):
        super().__init__()
        self.horizon_steps = horizon_steps
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, F)
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        return self.head(last).squeeze(-1)

    @torch.no_grad()
    def predict_with_uncertainty(
        self, x: torch.Tensor, n_samples: int = 8, dropout_at_inference: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor]:
        was_training = self.training
        if dropout_at_inference:
            self.train()
        preds = [self.forward(x) for _ in range(max(1, n_samples))]
        stack = torch.stack(preds, dim=0)
        mean = stack.mean(dim=0)
        std = stack.std(dim=0, unbiased=False).clamp_min(1e-6)
        if not was_training:
            self.eval()
        return mean, std

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state": self.state_dict(),
                "horizon_steps": self.horizon_steps,
            },
            path,
        )

    @classmethod
    def load(cls, path: Path, n_features: int, **kw) -> "GlucoseLSTM":
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        m = cls(n_features=n_features, horizon_steps=int(ckpt["horizon_steps"]), **kw)
        m.load_state_dict(ckpt["state"])
        m.eval()
        return m


def nan_rmse(pred: torch.Tensor, target: torch.Tensor) -> float:
    m = nn.MSELoss(reduction="mean")
    if pred.shape != target.shape:
        raise ValueError("shape mismatch")
    return math.sqrt(float(m(pred, target)))
