import mlflow
from lightning.pytorch.cli import LightningCLI


def lightning_cli():
    mlflow.autolog()
    cli = LightningCLI()


if __name__ == "__main__":
    lightning_cli()
