import torch
from .i_vector_function import IVectorFunction


class LogarithmFunction(IVectorFunction):
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nan_to_num(torch.log(x))

    @property
    def num_inputs(self) -> int:
        return 1

    @property
    def num_outputs(self) -> int:
        return 1
