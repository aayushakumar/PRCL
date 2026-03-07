"""MLP projection head for contrastive SSL."""

import torch.nn as nn


class ProjectionHead(nn.Module):
    """Two-layer MLP projection head (SimCLR-style).

    Maps backbone features to a lower-dimensional embedding space
    where the contrastive loss operates.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 2048, output_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)
