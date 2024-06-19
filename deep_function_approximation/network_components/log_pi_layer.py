import math
import torch
import torch.nn as nn


class LogPiLayer(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.scale = torch.tensor(output_dim, dtype=torch.float32).sqrt()
        weight = torch.randn((output_dim, input_dim))
        weight = weight * math.sqrt(2 / output_dim)
        diagonal = weight.diagonal()
        diagonal += 1
        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(torch.zeros(output_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip = x
        imaginary = torch.pi * (1.0 - torch.sign(x)) / 2.0
        x = x.abs().log()
        x = nn.functional.linear(x, self.weight, self.bias)
        x = x / self.scale
        x = x.exp().nan_to_num()
        # Handle the effect of the imaginary part
        # Note: since the bias is real, we don't add it here
        imaginary = nn.functional.linear(imaginary, self.weight)
        imaginary = imaginary / self.scale
        x = x * imaginary.cos()
        x = x + skip
        return x
