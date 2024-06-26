from functools import partial
import torch
import torch.nn as nn
from deep_function_approximation.network_components import LogSpaceConverter, init_weights_kaiming_normal


class MLPLogApproximator(nn.Module):
    def __init__(
        self,
        num_inputs: int,
        num_outputs: int,
        hidden_size: int,
        nonlin: nn.Module,
        depth: int = 3,
        init_weights: bool = True,
    ):
        super().__init__()
        layers = [
            nn.Linear(num_inputs, hidden_size),
            nonlin,
            nn.BatchNorm1d(hidden_size),
        ]
        for idx in range(depth - 2):
            layers.extend([
                nn.Linear(hidden_size, hidden_size),
                nonlin,
                nn.BatchNorm1d(hidden_size),
            ])
        layers.append(LogSpaceConverter(hidden_size, num_outputs, scalar_output=True))
        self.mlp = nn.Sequential(*layers)
        if init_weights:
            self.apply(partial(init_weights_kaiming_normal, module_types=[nn.Linear]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)
