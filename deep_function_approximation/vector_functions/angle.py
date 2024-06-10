import torch
from .i_vector_function import IVectorFunction


class AngleFunction(IVectorFunction):
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return torch.atan2(x[:, 0:1], x[:, 1:2])

    @property
    def num_inputs(self) -> int:
        return 2

    @property
    def num_outputs(self) -> int:
        return 1
