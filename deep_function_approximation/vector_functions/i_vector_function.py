from abc import ABC, abstractmethod
import torch


class IVectorFunction(ABC):

    @abstractmethod
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @property
    @abstractmethod
    def num_inputs(self) -> int:
        return 1

    @property
    @abstractmethod
    def num_outputs(self) -> int:
        return 1
