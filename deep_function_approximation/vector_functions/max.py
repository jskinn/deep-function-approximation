import torch
from .i_vector_function import IVectorFunction


class MaxFunction(IVectorFunction):
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        result = torch.max(x, dim=-1, keepdim=True)
        return result.values

    @property
    def num_inputs(self) -> None:
        return None

    @property
    def num_outputs(self) -> int:
        return 1
