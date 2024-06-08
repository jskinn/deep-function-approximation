import torch
from torch.utils.data import Dataset
from deep_function_approximation.vector_functions import IVectorFunction
from .vector_function_batch import VectorFunctionBatch


class NormalDistributionDataset(Dataset[VectorFunctionBatch]):
    def __init__(self, epoch_length: int, block_size: int, function: IVectorFunction):
        self.epoch_length = int(epoch_length)
        self.block_size = int(block_size)
        self.function = function

    def __len__(self) -> int:
        return self.epoch_length

    def __getitem__(self, item: int) -> VectorFunctionBatch:
        if not 0 <= item < len(self):
            raise IndexError(f"Index {item} out of range")
        x = torch.randn(self.block_size, self.function.num_inputs)
        y = self.function(x)
        return VectorFunctionBatch(x, y)
