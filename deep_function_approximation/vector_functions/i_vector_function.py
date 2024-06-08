from abc import ABC, abstractmethod
import torch


class IVectorFunction(ABC):
    @abstractmethod
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @property
    @abstractmethod
    def num_inputs(self) -> int | None:
        """The dimension of the input vector. If None, any size vector can be provided,
        configured in the dataset."""
        return 1

    @property
    @abstractmethod
    def num_outputs(self) -> int | None:
        """The dimension of the output vector. If None, output should be the same dimension as the input"""
        return 1
