from typing import Iterable, Type
import torch.nn as nn


def init_weights_kaiming_normal(
    module: nn.Module, module_types: Iterable[Type[nn.Module]]
):
    if any(isinstance(module, module_type) for module_type in module_types):
        if hasattr(module, "weight"):
            nn.init.kaiming_normal_(module.weight)
        if hasattr(module, "bias"):
            nn.init.constant_(module.bias, 0.0)
