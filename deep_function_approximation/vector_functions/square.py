import torch
from .i_vector_function import IVectorFunction


class SquareFunction(IVectorFunction):
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return x.square()

    @property
    def num_inputs(self) -> int:
        return 1

    @property
    def num_outputs(self) -> int:
        return 1
