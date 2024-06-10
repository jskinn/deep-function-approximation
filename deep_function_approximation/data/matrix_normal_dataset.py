import torch
from torch.utils.data import Dataset
from deep_function_approximation.matrix_functions import IMatrixFunction
from .function_batch import FunctionBatch


class MatrixNormalDataset(Dataset[FunctionBatch]):
    def __init__(
        self,
        epoch_length: int,
        block_size: int,
        function: IMatrixFunction,
        input_shape: tuple[int, int] = (2, 2),
        mean: float = 0.0,
        std: float = 1.0
    ):
        self.epoch_length = int(epoch_length)
        self.block_size = int(block_size)
        self.input_shape = input_shape
        self.mean = float(mean)
        self.std = float(std)
        self.function = function

    def __len__(self) -> int:
        return self.epoch_length

    def __getitem__(self, item: int) -> FunctionBatch:
        if not 0 <= item < len(self):
            raise IndexError(f"Index {item} out of range")
        input_shape = self.function.input_shape
        if input_shape is None:
            input_shape = self.input_shape
        x = self.std * torch.randn(self.block_size, input_shape[0], input_shape[1]) + self.mean
        y = self.function(x)
        return FunctionBatch(x, y)
