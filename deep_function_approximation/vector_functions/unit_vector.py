import torch
from .i_vector_function import IVectorFunction


class UnitVectorFunction(IVectorFunction):
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return x / torch.linalg.vector_norm(x, dim=-1, keepdim=True)

    @property
    def num_inputs(self) -> None:
        return None

    @property
    def num_outputs(self) -> None:
        return None
