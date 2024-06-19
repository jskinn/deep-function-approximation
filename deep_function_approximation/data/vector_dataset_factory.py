from typing import Any, Literal
from torch.utils.data import Dataset
from deep_function_approximation.vector_functions import IVectorFunction
from .function_batch import FunctionBatch
from .normal_distribution_dataset import NormalDistributionDataset
from .exponential_distribution_dataset import ExponentialDistributionDataset
from .uniform_distribution_dataset import UniformDistributionDataset


T_DatasetType = Literal["normal", "exponential", "uniform"]


def get_vector_dataset(
    distribution: T_DatasetType,
    function: IVectorFunction,
    block_size: int,
    epoch_length: int,
    vector_dim: int,
    dataset_kwargs: dict[str, Any] = None,
) -> Dataset[FunctionBatch]:
    distribution = str(distribution).lower()
    if dataset_kwargs is None:
        dataset_kwargs = {}
    if distribution == "exponential":
        return ExponentialDistributionDataset(
            function=function,
            block_size=block_size,
            epoch_length=epoch_length,
            vector_dim=vector_dim,
            **dataset_kwargs
        )
    elif distribution == "uniform":
        return UniformDistributionDataset(
            function=function,
            block_size=block_size,
            epoch_length=epoch_length,
            vector_dim=vector_dim,
            **dataset_kwargs
        )
    else:
        return NormalDistributionDataset(
            function=function,
            block_size=block_size,
            epoch_length=epoch_length,
            vector_dim=vector_dim,
            **dataset_kwargs
        )
