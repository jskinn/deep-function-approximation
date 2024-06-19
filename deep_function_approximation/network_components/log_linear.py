import torch
import torch.nn as nn


class LogLinear(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        log_x = (x.abs() + 1.0).log()
        x = torch.where(x > 0.0, log_x, x)
        return x
