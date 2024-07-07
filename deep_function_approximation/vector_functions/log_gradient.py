import torch
from .i_vector_function import IVectorFunction


class LogLinearGradientFunction(IVectorFunction):
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        max_scale = 1e4
        x1 = x[:, 0:1]
        y1 = x[:, 1:2]
        x2 = x[:, 2:3]
        y2 = x[:, 3:4]
        numerator = x2 - x1
        denominator = y2 - y1
        scale = numerator.abs().log().nan_to_num().clip_(
            min=-max_scale, max=max_scale
        ) - denominator.abs().log().nan_to_num().clip_(min=-max_scale, max=max_scale)
        sign = numerator.sign() * denominator.sign()
        return torch.cat([scale, sign], dim=-1)

    @property
    def num_inputs(self) -> int:
        return 4

    @property
    def num_outputs(self) -> int:
        return 2
