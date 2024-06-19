from typing import Literal
from pathlib import Path
import torch
import torch.nn as nn
from torchinfo import summary
from lightning import LightningModule
from lightning.pytorch.loggers import MLFlowLogger
import matplotlib.pyplot as plt
from deep_function_approximation.data import FunctionBatch


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
        self._vis_data = []
        torch.set_float32_matmul_precision(matmul_precision)

    @property
    def mlflow_logger(self) -> MLFlowLogger | None:
        return next(
            (logger for logger in self.loggers if isinstance(logger, MLFlowLogger)),
            None,
        )

    def on_fit_start(self) -> None:
        logger = self.mlflow_logger
        if logger is not None:
            summary_path = Path("model_summary.txt")
            with summary_path.open("w") as fp:
                fp.write(str(summary(self.network)))
            logger.experiment.log_artifact(logger.run_id, summary_path)
            summary_path.unlink()

    def training_step(self, batch: FunctionBatch, batch_idx):
        return self._step(batch, "Training")

    def validation_step(self, batch: FunctionBatch, batch_idx):
        return self._step(batch, "Validation", visualise=True)

    def test_step(self, batch: FunctionBatch, batch_idx):
        return self._step(batch, "Test", visualise=True)

    def on_validation_epoch_end(self) -> None:
        self._log_plots()

    def on_test_epoch_end(self) -> None:
        self._log_plots()

    def _step(
        self, batch: FunctionBatch, step_name: str = "Training", visualise: bool = False
    ):
        x = batch.x.reshape(-1, batch.x.size(-1))
        y = batch.y.reshape(-1, batch.y.size(-1))

        predictions = self.network(x)
        loss = self.loss_fn(predictions, y)
        self.log(f"{step_name} loss", loss, prog_bar=True)
        if visualise:
            self._vis_data.append((x, y, predictions.detach()))
        return loss

    def _log_plots(self) -> None:
        if len(self._vis_data) <= 0:
            return
        mlflow_logger = self.mlflow_logger
        if mlflow_logger is None:
            return
        x = torch.cat([batch_data[0].cpu() for batch_data in self._vis_data], dim=0)
        y = torch.cat([batch_data[1].cpu() for batch_data in self._vis_data], dim=0)
        preds = torch.cat([batch_data[2].cpu() for batch_data in self._vis_data], dim=0)
        self._vis_data = []

        num_inputs = x.size(-1)
        num_outputs = y.size(-1)
        fig, axes = plt.subplots(num_inputs, num_outputs, squeeze=False)
        for x_idx in range(num_inputs):
            for y_idx in range(num_outputs):
                y_range = min(3 * float(y[:, y_idx].std()), 10)
                axes[x_idx][y_idx].plot(
                    x[:, x_idx], y[:, y_idx], markersize=1, linestyle="none", marker="."
                )
                axes[x_idx][y_idx].plot(
                    x[:, x_idx],
                    preds[:, y_idx],
                    markersize=1,
                    linestyle="none",
                    marker=".",
                )
                axes[x_idx][y_idx].set_ylim(-1 * y_range, 1 * y_range)
        mlflow_logger.experiment.log_figure(
            mlflow_logger.run_id,
            fig,
            f"true_vs_predicted_epoch{self.current_epoch}.png",
        )

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.network.parameters(), lr=1e-4)
        return optimizer
