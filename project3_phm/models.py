"""Window regressors. Both keep dropout layers so they can be used for MC Dropout in step 4."""
import torch
from torch import nn

from prep import CAP


class CNN1D(nn.Module):
    def __init__(self, n_channels, width=64, dropout=0.2):
        super().__init__()

        def block(c_in, c_out, k):
            return nn.Sequential(nn.Conv1d(c_in, c_out, k, padding=k // 2), nn.ReLU(), nn.Dropout(dropout))

        self.features = nn.Sequential(block(n_channels, 32, 5), block(32, width, 5), block(width, width, 3))
        self.head = nn.Sequential(nn.Linear(2 * width, 64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, 1))

    def forward(self, x):
        """x: (B, window, channels) -> RUL (B,) in cycles."""
        h = self.features(x.transpose(1, 2))
        h = torch.cat([h.mean(-1), h[..., -1]], 1)          # whole-window summary + most recent step
        return self.head(h).squeeze(-1) * CAP


class LSTMRegressor(nn.Module):
    def __init__(self, n_channels, hidden=48, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(n_channels, hidden, num_layers=2, batch_first=True, dropout=dropout)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, 1))

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1]).squeeze(-1) * CAP


MODELS = {"cnn": CNN1D, "lstm": LSTMRegressor}


def n_params(m):
    return sum(p.numel() for p in m.parameters())
