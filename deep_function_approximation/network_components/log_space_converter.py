import torch
import torch.nn as nn


class LogSpaceConverter(nn.Module):
    """
    The theory is that function approximation is easier in absolute log space,
    using separate outputs for the value and sign.
    The final step of such an approximator then is to combine the outputs, as:
    y = sign(x[1]) * exp(x[0])
    Optionally, we include an additional passthrough output on x[2], as:
    y = sign(x[1]) * exp(x[0]) + x[2]

    This module takes an input tensor Bx(2*N) or Bx(3*N), depending on if the passthrough is used,
    and returns a BxN tensor.
    """

    def __init__(self, num_inputs: int, num_outputs: int, scalar_output: bool = False, max_exponent: float = 10.0):
        super().__init__()
        self.scalar_output = bool(scalar_output)
        self.max_exponent = abs(float(max_exponent))
        step = 3 if self.scalar_output else 2
        self.output_layer = nn.Linear(num_inputs, num_outputs * step)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        step = 3 if self.scalar_output else 2
        x = self.output_layer(x)
        num_outputs = x.size(-1)
        exponent = x[:, 0:num_outputs:step]
        sign = x[:, 1:num_outputs:step]
        exponent = exponent.clip(min=-self.max_exponent, max=self.max_exponent)
        exponent = exponent.exp().abs()
        sign = torch.tanh(sign)
        output = exponent * sign
        if self.scalar_output:
            output = output + x[:, 2:num_outputs:step]
        return output
