import torch
from .i_vector_function import IVectorFunction


class LogQuotientFunction(IVectorFunction):
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        x1 = x[:, 0:1]
        x2 = x[:, 1:2]
        sign = x1.sign() * x2.sign()
        scale = x1.abs().log().nan_to_num().clip_(min=-50) - x2.abs().log().nan_to_num().clip_(min=-50)
        return torch.cat([scale, sign], dim=-1)

    @property
    def num_inputs(self) -> int:
        return 2

    @property
    def num_outputs(self) -> int:
        return 2
