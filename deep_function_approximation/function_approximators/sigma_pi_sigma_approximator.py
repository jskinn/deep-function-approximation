from functools import partial
import torch
import torch.nn as nn
from deep_function_approximation.network_components import (
    LogPiLayer,
    init_weights_kaiming_normal,
)


class SPSApproximator(nn.Module):
    def __init__(
        self,
        num_inputs: int,
        num_outputs: int,
        hidden_size: int,
        init_weights: bool = True,
    ):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(hidden_size),
            LogPiLayer(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(hidden_size),
            nn.Linear(hidden_size, num_outputs),
        )
        if init_weights:
            self.apply(
                partial(
                    init_weights_kaiming_normal, module_types=[nn.Linear]
                )
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)
