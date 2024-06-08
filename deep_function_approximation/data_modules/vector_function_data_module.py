from typing import Any, Literal
from lightning import LightningDataModule
from torch.utils.data import DataLoader
from deep_function_approximation.vector_functions import IVectorFunction
from deep_function_approximation.data import ExponentialDistributionDataset, NormalDistributionDataset


class VectorFunctionDataModule(LightningDataModule):
    def __init__(
        self,
        function: IVectorFunction,
        batch_size: int = 64,
        block_size: int = 64,
        vector_dim: int = 1,
        num_workers: int = 8,
        training_batches: int = 128,
        validation_batches: int = 16,
        test_batches: int = 16,
        distribution: Literal["normal"] | Literal["exponential"] = "normal",
        dataset_kwargs: dict[str, Any] = None,
    ):
        super().__init__()
        self.function = function
        self.vector_dim = int(vector_dim)
        self.batch_size = int(batch_size)
        self.block_size = int(block_size)
        self.num_workers = int(num_workers)
        self.training_batches = int(training_batches)
        self.validation_batches = int(validation_batches)
        self.test_batches = int(test_batches)

        if dataset_kwargs is None:
            self._dataset_kwargs = {}
        else:
            self._dataset_kwargs = dataset_kwargs

        if distribution == "normal":
            self._dataset_type = NormalDistributionDataset
        elif distribution == "exponential":
            self._dataset_type = ExponentialDistributionDataset
        else:
            self._dataset_type = NormalDistributionDataset

    def train_dataloader(self):
        return DataLoader(
            self._dataset_type(
                function=self.function,
                block_size=self.block_size,
                epoch_length=self.training_batches * self.batch_size,
                vector_dim=self.vector_dim,
                **self._dataset_kwargs,
            ),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self._dataset_type(
                function=self.function,
                block_size=self.block_size,
                epoch_length=self.validation_batches * self.batch_size,
                vector_dim=self.vector_dim,
                **self._dataset_kwargs,
            ),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self._dataset_type(
                function=self.function,
                block_size=self.block_size,
                epoch_length=self.test_batches * self.batch_size,
                vector_dim=self.vector_dim,
                **self._dataset_kwargs,
            ),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )
