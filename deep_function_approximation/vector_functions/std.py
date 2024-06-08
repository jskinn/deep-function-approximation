import torch
from .i_vector_function import IVectorFunction


class StandardDeviationFunction(IVectorFunction):
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return torch.std(x, dim=-1, keepdim=True)

    @property
    def num_inputs(self) -> None:
        return None

    @property
    def num_outputs(self) -> int:
        return 1
