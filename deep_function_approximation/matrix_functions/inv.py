import torch
from .i_matrix_function import IMatrixFunction


class InverseFunction(IMatrixFunction):
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return torch.linalg.inv(x)

    @property
    def input_shape(self) -> None:
        return None

    @property
    def output_shape(self) -> None:
        return None
