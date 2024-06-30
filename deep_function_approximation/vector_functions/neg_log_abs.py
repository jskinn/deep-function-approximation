import torch
from .i_vector_function import IVectorFunction


class NegativeLogAbsFunction(IVectorFunction):
    # Note: this function is equivalent to the log of 1/|x|
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return -1 * x.abs().log().nan_to_num().clip_(min=-100)

    @property
    def num_inputs(self) -> int:
        return 1

    @property
    def num_outputs(self) -> int:
        return 1


class SignedNegativeLogAbsFunction(NegativeLogAbsFunction):
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        scale = super().__call__(x)
        sign = x.sign()
        return torch.cat([scale, sign], dim=-1)

    @property
    def num_outputs(self) -> int:
        return 2
