import torch
from .i_vector_function import IVectorFunction


class AbsoluteReciprocalFunction(IVectorFunction):
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return (1.0 / x.abs()).nan_to_num().clip_(min=-1e4, max=1e4)

    @property
    def num_inputs(self) -> int:
        return 1

    @property
    def num_outputs(self) -> int:
        return 1
