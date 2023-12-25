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

        # loggers = [
        #         pl.loggers.MLFlowLogger(
        #         experiment_name=cfg.artifacts.experiment_name,
        #         tracking_uri="file:./.logs/my-mlflow-logs",
        #     ),
        # ]

        trainer = pl.Trainer(max_epochs=self.config.train.epoch_count)
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
