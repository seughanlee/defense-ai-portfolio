"""Compact CNN for spectrogram classification (sized for embedded CPU inference)."""
import torch
from torch import nn


def _block(c_in, c_out, n_conv):
    layers = []
    for i in range(n_conv):
        layers += [nn.Conv2d(c_in if i == 0 else c_out, c_out, 3, padding=1, bias=False),
                   nn.BatchNorm2d(c_out), nn.ReLU(inplace=True)]
    return nn.Sequential(*layers, nn.MaxPool2d(2))


class MicroDopplerCNN(nn.Module):
    def __init__(self, n_classes, widths=(16, 32, 64, 96), dropout=0.3):
        super().__init__()
        c, blocks = 1, []
        for i, w in enumerate(widths):
            blocks.append(_block(c, w, 1 if i < 2 else 2))
            c = w
        self.features = nn.Sequential(*blocks)
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(dropout), nn.Linear(c, n_classes))

    def forward(self, x):
        """x: (B, 1, freq, time) in [0, 1] -> logits (B, n_classes)."""
        return self.head(self.features(x))


def n_params(model):
    return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    m = MicroDopplerCNN(5)
    print(n_params(m), m(torch.zeros(2, 1, 128, 128)).shape)
