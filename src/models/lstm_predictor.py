"""GlucoseLSTM: multivariate lag window → scalar prediction at a fixed horizon (normalized space)."""

import torch
import torch.nn as nn

# -----------------------------------------------------------------------------
# Defaults
# -----------------------------------------------------------------------------

_DEFAULT_HIDDEN = 64
_DEFAULT_NUM_LAYERS = 2
_DEFAULT_DROPOUT = 0.1
_MC_STD_EPS = 1e-6


# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------


class GlucoseLSTM(nn.Module):
    """
    Stack LSTM over ``(batch, lookback, n_features)``, take the last time step, MLP head → 1 dim.

    Training target is z-scored glucose at ``t + horizon_steps``; inference matches
    ``LstmForecastTool`` (checkpoint + optional MC dropout).
    """

    def __init__(
        self,
        n_features: int,
        hidden: int = _DEFAULT_HIDDEN,
        num_layers: int = _DEFAULT_NUM_LAYERS,
        dropout: float = _DEFAULT_DROPOUT,
        horizon_steps: int = 6,
    ) -> None:
        super().__init__()
        self.horizon_steps = horizon_steps
        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_dropout,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``x`` shape ``(B, L, F)``; returns ``(B,)`` scalar predictions."""
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        return self.head(last).squeeze(-1)

    @torch.no_grad()
    def predict_with_uncertainty(
        self,
        x: torch.Tensor,
        n_samples: int = 8,
        dropout_at_inference: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Multiple stochastic forwards (dropout on) → mean and std of predictions per batch element.

        Temporarily sets ``train()`` when ``dropout_at_inference``; restores previous mode in
        ``finally`` so callers always get eval state back if they started in eval.
        """
        n = max(1, int(n_samples))
        was_training = self.training
        try:
            if dropout_at_inference:
                self.train()
            preds = [self.forward(x) for _ in range(n)]
            stack = torch.stack(preds, dim=0)
            mean = stack.mean(dim=0)
            std = stack.std(dim=0, unbiased=False).clamp_min(_MC_STD_EPS)
            return mean, std
        finally:
            if not was_training:
                self.eval()
