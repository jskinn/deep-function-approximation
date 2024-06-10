from typing import NamedTuple
from torch import Tensor


class FunctionBatch(NamedTuple):
    x: Tensor
    y: Tensor
