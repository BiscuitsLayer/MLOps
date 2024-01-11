import os

import fire
import lightning.pytorch as pl
import torch
from dvc.repo import Repo

# Import parsed configs
from config.config_handler import all_config
from mnist.src.data import MNISTDatamodule
from mnist.src.model import MNISTModel


class Train:
    def __init__(self):
        torch.set_float32_matmul_precision("medium")
        self.config = all_config

        if not os.path.isdir(self.config.dataset.path) or not os.path.isdir(
            self.config.model.path
        ):
            # Pull repository data from Google Drive using DVC
            repo = Repo(".")
            repo.pull()

    def run_train(self):
        """
        Run model train on MNIST dataset
        """
        datamodule = MNISTDatamodule(self.config)
        model = MNISTModel(self.config)

        if self.config.train.load_from_file:
            model.load_state_dict(
                torch.load(
                    f"{self.config.model.path}/{self.config.model.pretrained_model_file}"
                )
            )
            print(
                f"Loaded from {self.config.model.pretrained_model_file} successfully!"
            )

        loggers = [
            pl.loggers.CSVLogger(
                "./.logs/csv-logs", name=self.config.artifacts.experiment_name
            ),
            pl.loggers.MLFlowLogger(
                tracking_uri="file:./.logs/mlflow-logs",
                experiment_name=self.config.artifacts.experiment_name,
            ),
        ]

        callbacks = [
            pl.callbacks.LearningRateMonitor(logging_interval="step"),
            pl.callbacks.DeviceStatsMonitor(),
        ]

        if self.config.artifacts.save_checkpoints:
            callbacks.append(
                pl.callbacks.ModelCheckpoint(
                    dirpath=self.config.model.path,
                    monitor="val_loss",
                    filename="model-{epoch:02d}-{val_loss:.2f}",
                    save_top_k=self.config.artifacts.top_k_value,
                    every_n_epochs=self.config.artifacts.every_n_epochs,
                    mode="min",
                )
            )

        trainer = pl.Trainer(
            accelerator=self.config.train.accelerator,
            max_epochs=self.config.train.epoch_count,
            logger=loggers,
            callbacks=callbacks,
            log_every_n_steps=1,
        )
        trainer.fit(model=model, datamodule=datamodule)

        torch.save(
            model.state_dict(),
            f"{self.config.model.path}/{self.config.model.new_model_file}",
        )
        print(f"Model saved to {self.config.model.new_model_file} successfully!")


def main():
    # Use fire to call Train class methods from CLI
    fire.Fire(Train)


if __name__ == "__main__":
    main()
