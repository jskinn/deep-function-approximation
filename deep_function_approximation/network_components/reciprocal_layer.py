import math
import torch
import torch.nn as nn


class ReciprocalLayer(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.numerator_layer = nn.Linear(dim, dim)
        self.denominator_layer = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        numerator: torch.Tensor = self.numerator_layer(x)
        denominator: torch.Tensor = self.denominator_layer(x)
        denominator = denominator.reciprocal().nan_to_num().clip(max=1e3, min=-1e3)
        x = x + numerator * denominator
        return x
