from typing import Literal
import torch
import torch.nn as nn
from lightning import LightningModule
from deep_function_approximation.data import FunctionBatch


class MatrixFunctionTrainingModule(LightningModule):
    def __init__(
        self,
        network: nn.Module,
        loss_function: nn.Module,
        flatten_input: bool = False,
        matmul_precision: Literal["medium", "high", "highest"] = "high",
    ):
        super().__init__()
        self.network = network
        self.loss_fn = loss_function
        self.flatten_input = bool(flatten_input)
        torch.set_float32_matmul_precision(matmul_precision)

    def training_step(self, batch: FunctionBatch, batch_idx):
        return self._step(batch, "Training")

    def validation_step(self, batch: FunctionBatch, batch_idx):
        return self._step(batch, "Validation")

    def test_step(self, batch: FunctionBatch, batch_idx):
        return self._step(batch, "Test")

    def _step(
        self,
        batch: FunctionBatch,
        step_name: str = "Training",
    ):
        x = batch.x
        y = batch.y
        if x.ndim > 3:
            x = x.reshape(-1, x.size(-2), x.size(-1))
        if y.ndim > 3:
            y = y.reshape(-1, y.size(-2), y.size(-1))
        if self.flatten_input:
            x = x.reshape(x.size(0), -1)
        predictions = self.network(x)
        if predictions.ndim < y.ndim:
            predictions = predictions.reshape(*y.shape)
        loss = self.loss_fn(predictions, y)
        self.log(f"{step_name} loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.network.parameters(), lr=1e-4)
        return optimizer
