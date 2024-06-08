import torch
from .i_vector_function import IVectorFunction


class CosineFunction(IVectorFunction):
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cos(x)

    @property
    def num_inputs(self) -> None:
        return 1

    @property
    def num_outputs(self) -> int:
        return 1
