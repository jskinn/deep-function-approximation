import torch
from torch.utils.data import Dataset
from deep_function_approximation.vector_functions import IVectorFunction
from .function_batch import FunctionBatch


class UniformDistributionDataset(Dataset[FunctionBatch]):
    def __init__(
        self,
        epoch_length: int,
        block_size: int,
        function: IVectorFunction,
        vector_dim: int = 1,
        min_value: float = -1.0,
        max_value: float = 1.0,
    ):
        self.epoch_length = int(epoch_length)
        self.block_size = int(block_size)
        self.vector_dim = int(vector_dim)
        self.function = function
        min_value = float(min_value)
        max_value = float(max_value)
        self.min_value = min(max_value, min_value)
        self.max_value = max(max_value, min_value)

    def __len__(self) -> int:
        return self.epoch_length

    def __getitem__(self, item: int) -> FunctionBatch:
        if not 0 <= item < len(self):
            raise IndexError(f"Index {item} out of range")
        vector_dim = self.function.num_inputs
        if vector_dim is None:
            vector_dim = self.vector_dim
        x = torch.empty(self.block_size, vector_dim).uniform_(self.min_value, self.max_value)
        y = self.function(x)
        return FunctionBatch(x, y)
