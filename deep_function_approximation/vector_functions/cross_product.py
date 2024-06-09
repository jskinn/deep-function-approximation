import torch
from .i_vector_function import IVectorFunction


class CrossProductFunction(IVectorFunction):
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        vector_1 = x[:, 0:3]
        vector_2 = x[:, 3:6]
        return torch.linalg.cross(vector_1, vector_2)

    @property
    def num_inputs(self) -> None:
        return 6

    @property
    def num_outputs(self) -> int:
        return 3
