import torch
import torch.nn as nn


class MLPApproximator(nn.Module):
    def __init__(
        self, num_inputs: int, num_outputs: int, hidden_size: int, nonlin: nn.Module
    ):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nonlin,
            nn.BatchNorm1d(hidden_size),
            nn.Linear(hidden_size, num_outputs),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)
