import torch
from .i_vector_function import IVectorFunction


class ReciprocalFunction(IVectorFunction):
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nan_to_num(1 / x).clip_(min=-1e4, max=1e4)

    @property
    def num_inputs(self) -> int:
        return 1

    @property
    def num_outputs(self) -> int:
        return 1
