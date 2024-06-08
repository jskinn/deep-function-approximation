import torch
from .i_vector_function import IVectorFunction


class ExponentialFunction(IVectorFunction):
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nan_to_num(torch.exp(x))

    @property
    def num_inputs(self) -> int:
        return 1

    @property
    def num_outputs(self) -> int:
        return 1
