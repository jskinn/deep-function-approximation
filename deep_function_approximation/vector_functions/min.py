import torch
from .i_vector_function import IVectorFunction


class MinFunction(IVectorFunction):
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        result = torch.min(x, dim=-1, keepdim=True)
        return result.values

    @property
    def num_inputs(self) -> None:
        return None

    @property
    def num_outputs(self) -> int:
        return 1
