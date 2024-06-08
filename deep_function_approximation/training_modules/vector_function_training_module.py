from typing import Literal
import torch
import torch.nn as nn
from lightning import LightningModule
from deep_function_approximation.data import VectorFunctionBatch


class VectorFunctionTrainingModule(LightningModule):
    def __init__(
        self,
        network: nn.Module,
        loss_function: nn.Module,
        matmul_precision: Literal["medium", "high", "highest"] = "high",
    ):
        super().__init__()
        self.network = network
        self.loss_fn = loss_function
        torch.set_float32_matmul_precision(matmul_precision)

    def training_step(self, batch: VectorFunctionBatch, batch_idx):
        return self._step(batch, "Training")

    def validation_step(self, batch: VectorFunctionBatch, batch_idx):
        return self._step(batch, "Validation")

    def test_step(self, batch: VectorFunctionBatch, batch_idx):
        return self._step(batch, "Test")

    def _step(
        self,
        batch: VectorFunctionBatch,
        step_name: str = "Training",
    ):
        x = batch.x.reshape(-1, batch.x.size(-1))
        y = batch.y.reshape(-1, batch.y.size(-1))

        predictions = self.network(x)
        loss = self.loss_fn(predictions, y)
        self.log(f"{step_name} loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.network.parameters(), lr=1e-4)
        return optimizer
