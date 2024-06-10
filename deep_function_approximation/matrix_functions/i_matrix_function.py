from abc import ABC, abstractmethod
import torch


class IMatrixFunction(ABC):
    """A function that operates on a single matrix"""

    @abstractmethod
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @property
    @abstractmethod
    def input_shape(self) -> tuple[int, int] | None:
        """The dimension of the input vector. If None, any size vector can be provided,
        configured in the dataset."""
        return 2, 2

    @property
    @abstractmethod
    def output_shape(self) -> tuple[int, int] | None:
        """The dimension of the output vector. If None, output should be the same dimension as the input"""
        return 2, 2
