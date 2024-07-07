import torch
from .i_vector_function import IVectorFunction


class LinearGradientFunction(IVectorFunction):
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        x1 = x[:, 0:1]
        y1 = x[:, 1:2]
        x2 = x[:, 2:3]
        y2 = x[:, 3:4]
        return torch.nan_to_num((y2 - y1) / (x2 - x1 + 4.0)).clip_(min=-1e4, max=1e4)

    @property
    def num_inputs(self) -> int:
        return 4

    @property
    def num_outputs(self) -> int:
        return 1
