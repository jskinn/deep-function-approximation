import torch
from torch.utils.data import Dataset
from deep_function_approximation.vector_functions import IVectorFunction
from .function_batch import FunctionBatch


class NormalDistributionDataset(Dataset[FunctionBatch]):
    def __init__(
        self,
        epoch_length: int,
        block_size: int,
        function: IVectorFunction,
        vector_dim: int = 1,
        mean: float = 0.0,
        std: float = 1.0
    ):
        self.epoch_length = int(epoch_length)
        self.block_size = int(block_size)
        self.vector_dim = int(vector_dim)
        self.mean = float(mean)
        self.std = float(std)
        self.function = function

    def __len__(self) -> int:
        return self.epoch_length

    def __getitem__(self, item: int) -> FunctionBatch:
        if not 0 <= item < len(self):
            raise IndexError(f"Index {item} out of range")
        vector_dim = self.function.num_inputs
        if vector_dim is None:
            vector_dim = self.vector_dim
        x = self.std * torch.randn(self.block_size, vector_dim) + self.mean
        y = self.function(x)
        return FunctionBatch(x, y)
