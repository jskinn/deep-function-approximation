from typing import NamedTuple
from torch import Tensor


class VectorFunctionBatch(NamedTuple):
    x: Tensor
    y: Tensor
